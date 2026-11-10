// Kinect SDK bridge for xenia-canary.
//
// Tracks the player with a real Kinect v1 through the Kinect for Windows SDK
// 1.8 (on the PC, natively) and sends the skeleton to Xenia over UDP, in the
// same "XKSK" packets as tools/kinect_bridge (the camera + MediaPipe
// bridge). Enable it in Xenia with kinect_skeleton_udp_port = 27050, and keep
// allow_kinect_camera = false: only one program can use the Kinect.
//
// The SDK's skeleton space is the one Xbox 360 titles use (metres, +x = the
// sensor's left, +y up, +z away from the sensor, same 20 joints in the same
// order). By default it is levelled with the SDK's floor plane and moved so
// that the floor is 0.8 m below the sensor, like Xenia's emulated sensor.
//
// Written for the C# 5 compiler that ships with Windows (see build.ps1).

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Microsoft.Kinect;

namespace XeniaKinectSdkBridge {

class Options {
  public int Port = 27050;
  public float Smoothing = 0.5f;
  public bool Level = true;
  public bool InvertX = false;
  public float EmulatedSensorHeight = 0.8f;
  public bool Seated = false;
  // The camera image (XKCL packets) for the games' player faces; every Nth
  // camera frame is sent (2 = 15 per second).
  public bool SendColour = true;
  public int ColourEvery = 2;
  // Send a square around the player's head instead of the whole camera image
  // (what the games show in the player slot); FaceSize is its width in metres.
  public bool FaceCrop = true;
  public float FaceSize = 0.30f;
  // The SDK's head joint sits near the top of the head; the crop is centred
  // this many metres below it so the face ends up in the middle.
  public float FaceOffset = 0.08f;
  public bool SendDepth = true;
  // Move the motor when the game asks (its auto-tilt at start-up).
  public bool AutoTilt = true;
  // Keep only the part of the SDK's player outline that is near the
  // player's skeleton (drops a bed or chair the SDK glued to the player).
  public bool CleanOutline = true;
  // How many people to send (the Kinect sees up to 6; full skeletons for 2
  // of them at a time). 1 sends the old one-player packets.
  public int Players = 6;
  // Let the game pick who gets the two full skeletons (Just Dance does).
  public bool GameChooses = true;

  public static Options Parse(string[] args) {
    Options o = new Options();
    for (int i = 0; i < args.Length; ++i) {
      string a = args[i].ToLowerInvariant();
      string next = i + 1 < args.Length ? args[i + 1] : null;
      if (a == "--port" && next != null) {
        o.Port = int.Parse(next, CultureInfo.InvariantCulture);
        ++i;
      } else if (a == "--smoothing" && next != null) {
        o.Smoothing = Math.Max(0f, Math.Min(0.95f,
            float.Parse(next, CultureInfo.InvariantCulture)));
        ++i;
      } else if (a == "--raw") {
        o.Level = false;
      } else if (a == "--invert-x") {
        o.InvertX = true;
      } else if (a == "--seated") {
        o.Seated = true;
      } else if (a == "--no-face-crop") {
        o.FaceCrop = false;
      } else if (a == "--face-size" && next != null) {
        o.FaceSize = Math.Max(0.2f, Math.Min(1.5f,
            float.Parse(next, CultureInfo.InvariantCulture)));
        ++i;
      } else if (a == "--face-offset" && next != null) {
        o.FaceOffset = Math.Max(-0.3f, Math.Min(0.3f,
            float.Parse(next, CultureInfo.InvariantCulture)));
        ++i;
      } else if (a == "--no-colour" || a == "--no-color") {
        o.SendColour = false;
      } else if ((a == "--colour-every" || a == "--color-every") &&
                 next != null) {
        o.ColourEvery = Math.Max(1, Math.Min(10,
            int.Parse(next, CultureInfo.InvariantCulture)));
        ++i;
      } else if (a == "--no-depth") {
        o.SendDepth = false;
      } else if (a == "--no-auto-tilt") {
        o.AutoTilt = false;
      } else if (a == "--raw-outline") {
        o.CleanOutline = false;
      } else if (a == "--players" && next != null) {
        o.Players = Math.Max(1, Math.Min(6,
            int.Parse(next, CultureInfo.InvariantCulture)));
        ++i;
      } else if (a == "--sdk-chooses") {
        o.GameChooses = false;
      } else if (a == "--help" || a == "-h" || a == "/?") {
        MessageBox.Show(
            "KinectSdkBridge [--port 27050] [--smoothing 0.5] [--raw] " +
            "[--invert-x] [--seated] [--no-depth]\n\n" +
            "--smoothing  0 = off (lowest latency) .. 0.95 = very smooth\n" +
            "--raw        don't level the skeleton with the floor\n" +
            "--invert-x   swap left and right\n" +
            "--seated     track the upper body only (Kinect seated mode)\n" +
            "--no-depth   send only the skeleton (Xenia draws the body)\n" +
            "--no-auto-tilt  ignore the game's tilt requests\n" +
            "--raw-outline   send the SDK's player outline as it is\n" +
            "--players 6     how many people to send (1 = one player only)\n" +
            "--sdk-chooses   the SDK, not the game, picks the two full " +
            "skeletons\n\n" +
            "In the window: Up/Down tilt the Kinect, O toggles the outline " +
            "clean-up, Q or Esc quits.",
            "Kinect SDK bridge");
        return null;
      }
    }
    return o;
  }
}

// One XKSK v1 packet: "XKSK", version 1, sequence, tracked, 20 joints as
// x, y, z floats (metres), 20 joint tracking states. Little-endian.
// XKSK v2 (everyone): "XKSK", version 2, sequence, count, then per body
// u32 tracking ID, u8 number (1-6, its value in the XKDP outline), u8 state
// (1 = position only, 2 = tracked), u8 primary, u8 0, 3 floats position,
// 60 floats joints, 20 joint states (280 bytes).
class Body {
  public int TrackingId;
  public int Number;
  public bool Tracked;
  public bool Primary;
  public float[] Position = new float[3];
  public float[] Joints = new float[Packet.JointCount * 3];
  public byte[] States = new byte[Packet.JointCount];
}

static class Packet {
  public const int JointCount = 20;
  public const int BodySize = 8 + 12 + JointCount * 13;

  public static byte[] BuildV2(uint sequence, List<Body> bodies) {
    byte[] p = new byte[16 + bodies.Count * BodySize];
    p[0] = (byte)'X';
    p[1] = (byte)'K';
    p[2] = (byte)'S';
    p[3] = (byte)'K';
    BitConverter.GetBytes((uint)2).CopyTo(p, 4);
    BitConverter.GetBytes(sequence).CopyTo(p, 8);
    BitConverter.GetBytes((uint)bodies.Count).CopyTo(p, 12);
    int o = 16;
    foreach (Body b in bodies) {
      BitConverter.GetBytes(b.TrackingId).CopyTo(p, o);
      p[o + 4] = (byte)b.Number;
      p[o + 5] = (byte)(b.Tracked ? 2 : 1);
      p[o + 6] = (byte)(b.Primary ? 1 : 0);
      for (int i = 0; i < 3; ++i) {
        BitConverter.GetBytes(b.Position[i]).CopyTo(p, o + 8 + i * 4);
      }
      if (b.Tracked) {
        for (int i = 0; i < JointCount * 3; ++i) {
          BitConverter.GetBytes(b.Joints[i]).CopyTo(p, o + 20 + i * 4);
        }
        Array.Copy(b.States, 0, p, o + 20 + JointCount * 12, JointCount);
      }
      o += BodySize;
    }
    return p;
  }

  public static byte[] Build(uint sequence, float[] joints, byte[] states) {
    byte[] p = new byte[16 + JointCount * 12 + JointCount];
    p[0] = (byte)'X';
    p[1] = (byte)'K';
    p[2] = (byte)'S';
    p[3] = (byte)'K';
    BitConverter.GetBytes((uint)1).CopyTo(p, 4);
    BitConverter.GetBytes(sequence).CopyTo(p, 8);
    BitConverter.GetBytes((uint)(joints != null ? 1 : 0)).CopyTo(p, 12);
    if (joints != null) {
      for (int i = 0; i < JointCount * 3; ++i) {
        BitConverter.GetBytes(joints[i]).CopyTo(p, 16 + i * 4);
      }
      Array.Copy(states, 0, p, 16 + JointCount * 12, JointCount);
    }
    return p;
  }
}

class BridgeForm : Form {
  const DepthImageFormat DepthFormat = DepthImageFormat.Resolution320x240Fps30;
  const int DepthWidth = 320;
  const int DepthHeight = 240;

  // Window colours for players 2-6 (player 1 is blue with a green
  // skeleton).
  static readonly Color[] PlayerColours = {
      Color.Black, Color.DeepSkyBlue, Color.FromArgb(0, 220, 90),
      Color.FromArgb(255, 150, 0), Color.FromArgb(255, 70, 110),
      Color.FromArgb(240, 230, 0), Color.FromArgb(150, 255, 230)};

  static readonly int[,] Bones = {
      {0, 1}, {1, 2}, {2, 3}, {2, 4}, {4, 5}, {5, 6}, {6, 7}, {2, 8},
      {8, 9}, {9, 10}, {10, 11}, {0, 12}, {12, 13}, {13, 14}, {14, 15},
      {0, 16}, {16, 17}, {17, 18}, {18, 19}};

  readonly Options options_;
  readonly UdpClient udp_ = new UdpClient();
  KinectSensor sensor_;
  Skeleton[] skeletons_ = new Skeleton[0];
  short[] depth_pixels_;
  int[] colour_pixels_ = new int[DepthWidth * DepthHeight];
  Bitmap depth_bitmap_ = new Bitmap(DepthWidth, DepthHeight,
                                    PixelFormat.Format32bppRgb);
  int tracking_id_ = -1;
  uint sequence_ = 0;
  // Last known sensor height (floor plane D), used while the floor isn't
  // visible.
  float sensor_height_ = 0f;
  float[] floor_ = null;
  PointF[] drawn_joints_ = null;
  byte[] drawn_states_ = null;
  string status_ = "Looking for a Kinect...";
  DateTime last_tilt_ = DateTime.MinValue;
  int skeleton_frames_ = 0;
  DateTime rate_start_ = DateTime.Now;
  double rate_ = 0;

  public BridgeForm(Options options) {
    options_ = options;
    Text = "Kinect SDK bridge -> Xenia (UDP " + options.Port + ")";
    ClientSize = new Size(DepthWidth * 2, DepthHeight * 2 + 60);
    DoubleBuffered = true;
    BackColor = Color.Black;
    KeyPreview = true;
    udp_.Connect("127.0.0.1", options.Port);
    try {
      // Without this, Windows reports "port unreachable" (Xenia not running
      // yet) as an error on the next receive.
      const int SIO_UDP_CONNRESET = -1744830452;
      udp_.Client.IOControl(SIO_UDP_CONNRESET, new byte[] {0, 0, 0, 0}, null);
    } catch (Exception) {
    }
    System.Threading.Thread receiver =
        new System.Threading.Thread(ReceiveFromXenia);
    receiver.IsBackground = true;
    receiver.Start();
    tilt_timer_ = new System.Threading.Timer(TiltTick, null, 500, 250);
    KinectSensor.KinectSensors.StatusChanged += OnSensorStatusChanged;
    Shown += delegate { StartFirstSensor(); };
    FormClosed += delegate {
      tilt_timer_.Dispose();
      StopSensor();
    };
    KeyDown += OnKey;
  }

  void StartFirstSensor() {
    foreach (KinectSensor sensor in KinectSensor.KinectSensors) {
      if (sensor.Status == KinectStatus.Connected) {
        StartSensor(sensor);
        return;
      }
    }
    if (KinectSensor.KinectSensors.Count == 0) {
      status_ = "No Kinect found. Plug it in (with its power adapter).";
    } else {
      status_ = "Kinect status: " + KinectSensor.KinectSensors[0].Status +
                " (needs power adapter / SDK driver)";
    }
    Invalidate();
  }

  void OnSensorStatusChanged(object sender, StatusChangedEventArgs e) {
    if (sensor_ == null && e.Status == KinectStatus.Connected) {
      StartSensor(e.Sensor);
    } else if (sensor_ == e.Sensor && e.Status != KinectStatus.Connected) {
      StopSensor();
      status_ = "Kinect disconnected (" + e.Status + ")";
      SendNobody();
      Invalidate();
    }
  }

  void StartSensor(KinectSensor sensor) {
    try {
      if (options_.Smoothing > 0f) {
        TransformSmoothParameters smoothing = new TransformSmoothParameters();
        smoothing.Smoothing = options_.Smoothing;
        smoothing.Correction = 0.5f;
        smoothing.Prediction = 0.5f;
        smoothing.JitterRadius = 0.05f;
        smoothing.MaxDeviationRadius = 0.04f;
        sensor.SkeletonStream.Enable(smoothing);
      } else {
        sensor.SkeletonStream.Enable();
      }
      if (options_.Seated) {
        sensor.SkeletonStream.TrackingMode = SkeletonTrackingMode.Seated;
      }
      sensor.DepthStream.Enable(DepthFormat);
      depth_pixels_ = new short[sensor.DepthStream.FramePixelDataLength];
      if (options_.SendColour) {
        sensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);
        video_pixels_ = new byte[sensor.ColorStream.FramePixelDataLength];
        video_work_ = new byte[video_pixels_.Length];
        sensor.ColorFrameReady += OnColourFrame;
      }
      sensor.SkeletonFrameReady += OnSkeletonFrame;
      sensor.DepthFrameReady += OnDepthFrame;
      sensor.Start();
      sensor_ = sensor;
      status_ = "Kinect running. Stand 2-3 m in front of it.";
    } catch (IOException) {
      status_ = "The Kinect is in use by another program. Close it (in " +
                "Xenia set allow_kinect_camera = false) and restart this.";
    } catch (Exception ex) {
      status_ = "Cannot start the Kinect: " + ex.Message;
    }
    Invalidate();
  }

  void StopSensor() {
    if (sensor_ == null) {
      return;
    }
    KinectSensor sensor = sensor_;
    sensor_ = null;
    sensor.SkeletonFrameReady -= OnSkeletonFrame;
    sensor.DepthFrameReady -= OnDepthFrame;
    sensor.ColorFrameReady -= OnColourFrame;
    try {
      sensor.Stop();
    } catch (Exception) {
    }
    SendNobody();
  }

  void SendNobody() {
    byte[] packet = options_.Players > 1
                        ? Packet.BuildV2(++sequence_, new List<Body>())
                        : Packet.Build(++sequence_, null, null);
    try {
      udp_.Send(packet, packet.Length);
    } catch (SocketException) {
      // Xenia not running yet: nothing listens on the port.
    }
  }

  void OnKey(object sender, KeyEventArgs e) {
    if (e.KeyCode == Keys.Q || e.KeyCode == Keys.Escape) {
      Close();
    } else if (e.KeyCode == Keys.O) {
      options_.CleanOutline = !options_.CleanOutline;
      Invalidate();
    } else if (e.KeyCode == Keys.Z || e.KeyCode == Keys.X) {
      // Face crop: Z zooms in, X zooms out.
      options_.FaceSize = Math.Max(0.15f, Math.Min(1.0f,
          options_.FaceSize + (e.KeyCode == Keys.X ? 0.02f : -0.02f)));
      Invalidate();
    } else if (e.KeyCode == Keys.W || e.KeyCode == Keys.S) {
      // Face crop: W moves the face up in the square, S moves it down.
      options_.FaceOffset = Math.Max(-0.3f, Math.Min(0.3f,
          options_.FaceOffset + (e.KeyCode == Keys.W ? 0.01f : -0.01f)));
      SendFaceSettings();
      Invalidate();
    } else if ((e.KeyCode == Keys.Up || e.KeyCode == Keys.Down) &&
               sensor_ != null) {
      // Queued; TiltTick moves the motor within its rate limits.
      int basis = tilt_target_ != NoTarget ? tilt_target_ : tilt_angle_;
      tilt_target_ = Math.Max(-27, Math.Min(27,
          basis + (e.KeyCode == Keys.Up ? 3 : -3)));
      Invalidate();
    }
  }

  // Tilt. Xenia passes the game's XamNuiCameraElevationSetAngle requests as
  // "XKTR", u32 version 1, i32 angle (degrees, + = up); the bridge answers
  // with "XKTS", u32 version 1, i32 angle, u32 flags (1 = moving or a move
  // is queued), 3 floats accelerometer, about twice a second. Little-endian.
  // The motor is moved at most once every 1.2 s and 15 times in 2 minutes
  // (the SDK's guidance: more wears it out and it locks itself for a while).
  const int NoTarget = int.MinValue;
  System.Threading.Timer tilt_timer_;
  int tilt_busy_ = 0;
  int tilt_target_ = int.MinValue;  // NoTarget (atomic int; Interlocked)
  volatile int tilt_angle_ = 0;
  volatile bool tilt_known_ = false;
  volatile bool tilt_moving_ = false;
  volatile bool tilt_waiting_ = false;
  volatile int game_tilt_ = int.MinValue;     // last angle the game asked for
  DateTime tilt_read_ = DateTime.MinValue;
  DateTime tilt_sent_ = DateTime.MinValue;
  Queue<DateTime> tilt_moves_ = new Queue<DateTime>();
  string tilt_error_;

  void ReceiveFromXenia() {
    IPEndPoint from = null;
    while (true) {
      byte[] p;
      try {
        p = udp_.Receive(ref from);
      } catch (ObjectDisposedException) {
        return;
      } catch (SocketException) {
        System.Threading.Thread.Sleep(200);
        continue;
      }
      if (p.Length >= 16 && p[0] == 'X' && p[1] == 'K' && p[2] == 'C' &&
          p[3] == 'H' && BitConverter.ToUInt32(p, 4) == 1) {
        chosen_first_ = BitConverter.ToInt32(p, 8);
        chosen_second_ = BitConverter.ToInt32(p, 12);
        choice_pending_ = true;
        continue;
      }
      if (p.Length >= 12 && p[0] == 'X' && p[1] == 'K' && p[2] == 'T' &&
          p[3] == 'R' && BitConverter.ToUInt32(p, 4) == 1) {
        int angle = BitConverter.ToInt32(p, 8);
        if (angle < -27 || angle > 27) {
          continue;
        }
        game_tilt_ = angle;
        if (options_.AutoTilt) {
          tilt_target_ = angle;
        }
      }
    }
  }

  bool MotorMayMove(DateTime now) {
    while (tilt_moves_.Count > 0 &&
           (now - tilt_moves_.Peek()).TotalSeconds > 120) {
      tilt_moves_.Dequeue();
    }
    return (now - last_tilt_).TotalSeconds >= 1.2 && tilt_moves_.Count < 15;
  }

  void TiltTick(object state) {
    if (System.Threading.Interlocked.Exchange(ref tilt_busy_, 1) == 1) {
      return;  // a move is still in progress
    }
    try {
      KinectSensor sensor = sensor_;
      if (sensor == null) {
        tilt_known_ = false;
        return;
      }
      DateTime now = DateTime.Now;
      if (!tilt_known_ || (now - tilt_read_).TotalSeconds >= 2) {
        tilt_angle_ = sensor.ElevationAngle;
        tilt_known_ = true;
        tilt_read_ = now;
      }
      int requested = tilt_target_;
      tilt_waiting_ = false;
      if (requested != NoTarget) {
        int target = Math.Max(sensor.MinElevationAngle,
                              Math.Min(sensor.MaxElevationAngle, requested));
        if (Math.Abs(target - tilt_angle_) <= 1) {
          System.Threading.Interlocked.CompareExchange(ref tilt_target_,
                                                       NoTarget, requested);
        } else if (MotorMayMove(now)) {
          tilt_moving_ = true;
          SendTiltStatus();
          sensor.ElevationAngle = target;  // returns when the move is done
          last_tilt_ = DateTime.Now;
          tilt_moves_.Enqueue(last_tilt_);
          tilt_angle_ = sensor.ElevationAngle;
          tilt_read_ = last_tilt_;
          // One attempt per request, also when the motor can't get there.
          System.Threading.Interlocked.CompareExchange(ref tilt_target_,
                                                       NoTarget, requested);
          tilt_moving_ = false;
          tilt_sent_ = DateTime.MinValue;
        } else {
          tilt_waiting_ = true;
        }
      }
      if ((now - tilt_sent_).TotalSeconds >= 0.5) {
        SendTiltStatus();
      }
    } catch (Exception ex) {
      tilt_error_ = ex.GetType().Name + ": " + ex.Message;
      tilt_moving_ = false;
      tilt_target_ = NoTarget;
    } finally {
      System.Threading.Interlocked.Exchange(ref tilt_busy_, 0);
    }
  }

  void SendTiltStatus() {
    tilt_sent_ = DateTime.Now;
    if (!tilt_known_) {
      return;
    }
    float gx = 0f, gy = 0f, gz = 0f;
    try {
      KinectSensor sensor = sensor_;
      if (sensor != null) {
        Vector4 g = sensor.AccelerometerGetCurrentReading();
        gx = g.X;
        gy = g.Y;
        gz = g.Z;
      }
    } catch (Exception) {
    }
    bool busy = tilt_moving_ || tilt_target_ != NoTarget;
    byte[] p = new byte[28];
    p[0] = (byte)'X';
    p[1] = (byte)'K';
    p[2] = (byte)'T';
    p[3] = (byte)'S';
    BitConverter.GetBytes((uint)1).CopyTo(p, 4);
    BitConverter.GetBytes(tilt_angle_).CopyTo(p, 8);
    BitConverter.GetBytes((uint)(busy ? 1 : 0)).CopyTo(p, 12);
    BitConverter.GetBytes(gx).CopyTo(p, 16);
    BitConverter.GetBytes(gy).CopyTo(p, 20);
    BitConverter.GetBytes(gz).CopyTo(p, 24);
    try {
      udp_.Send(p, p.Length);
    } catch (SocketException) {
    }
  }

  string TiltText() {
    string text = "   Tilt: " + (tilt_known_ ? tilt_angle_.ToString(
                                     CultureInfo.InvariantCulture) : "?") +
                  " deg (Up/Down)";
    int game = game_tilt_;
    if (game != int.MinValue) {
      text += ", game asked " + game.ToString(CultureInfo.InvariantCulture) +
              (options_.AutoTilt ? "" : " (ignored: --no-auto-tilt)");
    }
    if (tilt_moving_) {
      text += ", moving";
    } else if (tilt_waiting_) {
      text += ", waiting (motor rest)";
    }
    if (tilt_error_ != null) {
      text += ", tilt error: " + tilt_error_;
    }
    return text;
  }

  // Index of the skeleton to send: the one sent last while it is still
  // tracked, otherwise the tracked one nearest to the middle of the view.
  // With more than one player, the primary player stays the same person
  // while they are in view (the game may give their full skeleton to
  // someone else for a while); null is returned meanwhile.
  Skeleton ChoosePlayer() {
    Skeleton best = null;
    foreach (Skeleton s in skeletons_) {
      if (s != null && s.TrackingState == SkeletonTrackingState.Tracked &&
          s.TrackingId == tracking_id_) {
        return s;
      }
    }
    if (options_.Players > 1 && tracking_id_ != -1) {
      foreach (Skeleton s in skeletons_) {
        if (s != null &&
            s.TrackingState == SkeletonTrackingState.PositionOnly &&
            s.TrackingId == tracking_id_) {
          return null;
        }
      }
    }
    foreach (Skeleton s in skeletons_) {
      if (s == null || s.TrackingState != SkeletonTrackingState.Tracked) {
        continue;
      }
      if (best == null || Math.Abs(s.Position.X) < Math.Abs(best.Position.X)) {
        best = s;
      }
    }
    tracking_id_ = best != null ? best.TrackingId : -1;
    return best;
  }

  void OnSkeletonFrame(object sender, SkeletonFrameReadyEventArgs e) {
    using (SkeletonFrame frame = e.OpenSkeletonFrame()) {
      if (frame == null) {
        return;
      }
      if (skeletons_.Length != frame.SkeletonArrayLength) {
        skeletons_ = new Skeleton[frame.SkeletonArrayLength];
      }
      frame.CopySkeletonDataTo(skeletons_);
      Tuple<float, float, float, float> plane = frame.FloorClipPlane;
      floor_ = null;
      if (plane != null) {
        float a = plane.Item1, b = plane.Item2, c = plane.Item3;
        float length = (float)Math.Sqrt(a * a + b * b + c * c);
        if (length > 0.5f && plane.Item4 > 0.1f) {
          floor_ = new float[] {a / length, b / length, c / length};
          sensor_height_ = plane.Item4;
        }
      }
    }

    ++skeleton_frames_;
    double elapsed = (DateTime.Now - rate_start_).TotalSeconds;
    if (elapsed >= 1.0) {
      rate_ = skeleton_frames_ / elapsed;
      skeleton_frames_ = 0;
      rate_start_ = DateTime.Now;
    }

    ApplyGameChoice();
    if (skeleton_frames_ % 15 == 1) {
      SendFaceSettings();
    }
    Skeleton player = ChoosePlayer();
    NumberPeople(player);
    drawn_others_.Clear();
    if (player == null) {
      face_found_ = false;
      drawn_joints_ = null;
      if (options_.Players == 1 || people_ == 0) {
        SendNobody();
        Invalidate();
        return;
      }
    }

    float[] joints = null;
    byte[] states = null;
    if (player != null) {
      joints = new float[Packet.JointCount * 3];
      states = new byte[Packet.JointCount];
      drawn_joints_ = new PointF[Packet.JointCount];
      drawn_states_ = states;
      for (int i = 0; i < Packet.JointCount; ++i) {
        Joint joint = player.Joints[(JointType)i];
        SkeletonPoint p = joint.Position;
        states[i] = (byte)joint.TrackingState;  // 0 not tracked, 1 inferred, 2
        DepthImagePoint d = sensor_.CoordinateMapper.MapSkeletonPointToDepthPoint(
            p, DepthFormat);
        drawn_joints_[i] = new PointF(d.X, d.Y);
        if (i == (int)JointType.Head) {
          // For the face crop: the head in depth-image pixels and its
          // distance.
          face_depth_point_ = new PointF(d.X, d.Y);
          face_z_ = p.Z;
          face_found_ = joint.TrackingState != JointTrackingState.NotTracked;
        }
        SendPoint(p, joints, i * 3);
      }
    }
    byte[] packet;
    if (options_.Players == 1) {
      packet = Packet.Build(++sequence_, joints, states);
    } else {
      List<Body> bodies = new List<Body>();
      for (int k = 0; k < skeletons_.Length; ++k) {
        Skeleton sk = skeletons_[k];
        int number = k + 1 < number_of_index_.Length ? number_of_index_[k + 1]
                                                      : 0;
        if (sk == null || number == 0) {
          continue;
        }
        Body b = new Body();
        b.TrackingId = sk.TrackingId;
        b.Number = number;
        b.Primary = sk == player;
        b.Tracked = sk.TrackingState == SkeletonTrackingState.Tracked;
        SendPoint(sk.Position, b.Position, 0);
        if (b.Tracked) {
          if (sk == player) {
            Array.Copy(joints, b.Joints, joints.Length);
            Array.Copy(states, b.States, states.Length);
          } else {
            PointF[] drawn = new PointF[Packet.JointCount];
            for (int i = 0; i < Packet.JointCount; ++i) {
              Joint joint = sk.Joints[(JointType)i];
              b.States[i] = (byte)joint.TrackingState;
              SendPoint(joint.Position, b.Joints, i * 3);
              DepthImagePoint d =
                  sensor_.CoordinateMapper.MapSkeletonPointToDepthPoint(
                      joint.Position, DepthFormat);
              drawn[i] = new PointF(d.X, d.Y);
            }
            drawn_others_.Add(new DrawnBody(drawn, b.States, number));
          }
        } else {
          // Position only: just the number, about head height.
          SkeletonPoint top = sk.Position;
          top.Y += 0.55f;
          DepthImagePoint d =
              sensor_.CoordinateMapper.MapSkeletonPointToDepthPoint(
                  top, DepthFormat);
          drawn_others_.Add(new DrawnBody(
              new PointF[] {new PointF(d.X, d.Y)}, null, number));
        }
        bodies.Add(b);
      }
      packet = Packet.BuildV2(++sequence_, bodies);
    }
    try {
      udp_.Send(packet, packet.Length);
    } catch (SocketException) {
    }
    Invalidate();
  }

  // A point as sent to Xenia: levelled (Transform), x flipped with
  // --invert-x, written to out[o..o+2].
  void SendPoint(SkeletonPoint p, float[] out_xyz, int o) {
    float[] v = Transform(p.X, p.Y, p.Z);
    out_xyz[o] = options_.InvertX ? -v[0] : v[0];
    out_xyz[o + 1] = v[1];
    out_xyz[o + 2] = v[2];
  }

  // Tells Xenia whether the camera image is the whole picture (the games
  // crop each player's face themselves) and the face offset (W/S), about
  // twice a second: "XKFO", u32 version 1, u32 flags (1 = whole image),
  // f32 offset in metres.
  void SendFaceSettings() {
    byte[] p = new byte[16];
    p[0] = (byte)'X';
    p[1] = (byte)'K';
    p[2] = (byte)'F';
    p[3] = (byte)'O';
    BitConverter.GetBytes((uint)1).CopyTo(p, 4);
    bool whole = !options_.FaceCrop || colour_error_ != null;
    BitConverter.GetBytes((uint)(whole ? 1 : 0)).CopyTo(p, 8);
    BitConverter.GetBytes(options_.FaceOffset).CopyTo(p, 12);
    try {
      udp_.Send(p, p.Length);
    } catch (SocketException) {
    }
  }

  // Everyone else's skeleton for the window.
  class DrawnBody {
    public PointF[] Joints;
    public byte[] States;
    public int Number;
    public DrawnBody(PointF[] joints, byte[] states, int number) {
      Joints = joints;
      States = states;
      Number = number;
    }
  }
  List<DrawnBody> drawn_others_ = new List<DrawnBody>();

  // Player numbers. A number is the body's skeleton slot + 1 in Xenia and
  // its value in the outline, so it is the SDK's own (slot + 1): it stays
  // the same for a person while they are in view, as on a console, also
  // when the game moves the two full skeletons around. The primary player
  // (the one the face crop and single-player features follow) is sent
  // first; with --players N only N people are sent, and with --players 1
  // the primary player is number 1 (the one-player packets).
  // number_of_index_[SDK player index] (index = SDK slot + 1); 0 = not sent.
  int[] number_of_index_ = new int[7];
  int people_ = 0;  // how many are sent
  int full_skeletons_ = 0;
  int primary_number_ = 1;

  void NumberPeople(Skeleton player) {
    Array.Clear(number_of_index_, 0, number_of_index_.Length);
    people_ = 0;
    full_skeletons_ = 0;
    int count = 0;
    if (player != null) {
      for (int k = 0; k < skeletons_.Length && k < 6; ++k) {
        if (skeletons_[k] == player) {
          number_of_index_[k + 1] = options_.Players == 1 ? 1 : k + 1;
          primary_number_ = number_of_index_[k + 1];
          ++count;
        }
      }
    }
    for (int k = 0; k < skeletons_.Length && k < 6; ++k) {
      Skeleton sk = skeletons_[k];
      if (sk == null || sk == player ||
          sk.TrackingState == SkeletonTrackingState.NotTracked ||
          count >= options_.Players) {
        continue;
      }
      number_of_index_[k + 1] = k + 1;
      ++count;
    }
    for (int k = 0; k < skeletons_.Length && k < 6; ++k) {
      if (number_of_index_[k + 1] != 0) {
        ++people_;
        if (skeletons_[k].TrackingState == SkeletonTrackingState.Tracked) {
          ++full_skeletons_;
        }
      }
    }
  }

  // The game's choice of who gets the two full skeletons (Xenia passes its
  // NuiSkeletonSetTrackedSkeletons calls as XKCH: "XKCH", u32 version 1,
  // u32 tracking ID x2, 0 = none). The SDK tracks the game's picks that are
  // in view; a free place goes to someone else in view (the main player
  // first, then whoever is nearest the middle), so with two people both
  // always have a full skeleton even when the game names only one.
  volatile int chosen_first_ = 0;
  volatile int chosen_second_ = 0;
  volatile bool choice_pending_ = false;
  bool app_chooses_ = false;
  int applied_first_ = -1;
  int applied_second_ = -1;
  string choice_text_ = "";

  static bool InView(Skeleton sk) {
    return sk != null && sk.TrackingState != SkeletonTrackingState.NotTracked;
  }

  void ApplyGameChoice() {
    KinectSensor sensor = sensor_;
    if (sensor == null || options_.Players == 1 || !options_.GameChooses) {
      return;
    }
    if (choice_pending_) {
      choice_pending_ = false;
      app_chooses_ = true;
    }
    if (!app_chooses_) {
      return;  // the SDK picks until the game does
    }
    int game_first = chosen_first_, game_second = chosen_second_;
    List<int> pick = new List<int>();
    foreach (Skeleton sk in skeletons_) {
      if (InView(sk) && sk.TrackingId != 0 &&
          (sk.TrackingId == game_first || sk.TrackingId == game_second) &&
          !pick.Contains(sk.TrackingId)) {
        pick.Add(sk.TrackingId);
      }
    }
    int games = pick.Count;
    if (pick.Count < 2) {
      foreach (Skeleton sk in skeletons_) {
        if (InView(sk) && sk.TrackingId == tracking_id_ &&
            !pick.Contains(sk.TrackingId)) {
          pick.Add(sk.TrackingId);
        }
      }
    }
    while (pick.Count < 2) {
      Skeleton best = null;
      foreach (Skeleton sk in skeletons_) {
        if (InView(sk) && !pick.Contains(sk.TrackingId) &&
            (best == null || Math.Abs(sk.Position.X) < Math.Abs(best.Position.X))) {
          best = sk;
        }
      }
      if (best == null) {
        break;
      }
      pick.Add(best.TrackingId);
    }
    int first = pick.Count > 0 ? pick[0] : 0;
    int second = pick.Count > 1 ? pick[1] : 0;
    if (first == applied_first_ && second == applied_second_) {
      return;
    }
    try {
      if (!sensor.SkeletonStream.AppChoosesSkeletons) {
        sensor.SkeletonStream.AppChoosesSkeletons = true;
      }
      if (second != 0) {
        sensor.SkeletonStream.ChooseSkeletons(first, second);
      } else if (first != 0) {
        sensor.SkeletonStream.ChooseSkeletons(first);
      } else {
        sensor.SkeletonStream.ChooseSkeletons();
      }
      applied_first_ = first;
      applied_second_ = second;
      choice_text_ = "game picked " + game_first +
                     (game_second != 0 ? " and " + game_second : "") +
                     (pick.Count > games ? ", also tracking " +
                                               (games == 0 ? first : second)
                                         : "");
    } catch (Exception ex) {
      choice_text_ = "choice failed: " + ex.Message;
    }
  }

  // Levels a point with the floor plane (rotating the floor normal to +y)
  // and moves it so that the floor is EmulatedSensorHeight below the sensor.
  float[] Transform(float x, float y, float z) {
    if (!options_.Level) {
      return new float[] {x, y, z};
    }
    if (floor_ != null) {
      // Rotation taking n to (0, 1, 0) (Rodrigues, axis n x up).
      float nx = floor_[0], ny = floor_[1], nz = floor_[2];
      float kx = -nz, kz = nx;  // axis n x (0, 1, 0) = (-nz, 0, nx)
      float s = (float)Math.Sqrt(kx * kx + kz * kz);  // sin(angle)
      float c = ny;                                    // cos(angle)
      if (s > 1e-6f) {
        kx /= s;
        kz /= s;
        // v' = v c + (k x v) s + k (k . v)(1 - c), with k = (kx, 0, kz).
        float kdotv = kx * x + kz * z;
        float cx = -kz * y, cy = kz * x - kx * z, cz = kx * y;  // k x v
        float rx = x * c + cx * s + kx * kdotv * (1 - c);
        float ry = y * c + cy * s;
        float rz = z * c + cz * s + kz * kdotv * (1 - c);
        x = rx;
        y = ry;
        z = rz;
      }
    }
    if (sensor_height_ > 0f) {
      y += sensor_height_ - options_.EmulatedSensorHeight;
    }
    return new float[] {x, y, z};
  }

  void OnDepthFrame(object sender, DepthImageFrameReadyEventArgs e) {
    using (DepthImageFrame frame = e.OpenDepthImageFrame()) {
      if (frame == null || frame.Width != DepthWidth ||
          frame.Height != DepthHeight) {
        return;
      }
      frame.CopyPixelDataTo(depth_pixels_);
    }
    BuildOutline();
    for (int i = 0; i < colour_pixels_.Length; ++i) {
      int raw = depth_pixels_[i];
      int mm = (raw >> DepthImageFrame.PlayerIndexBitmaskWidth) & 0x1FFF;
      int grey = mm <= 0 ? 0 : 255 - Math.Min(255, mm * 255 / 4500);
      int number = outline_[i];
      if (number == 1) {
        // Sent to Xenia as player 1.
        colour_pixels_[i] = (grey / 3 << 16) | (Math.Min(255, grey + 60) << 8) |
                            Math.Min(255, grey + 120);
      } else if (number >= 2 && number <= 6) {
        // Players 2-6, each in their own colour.
        int tint = PlayerColours[number].ToArgb();
        int r = (tint >> 16) & 0xFF, gr = (tint >> 8) & 0xFF, b = tint & 0xFF;
        colour_pixels_[i] = (Math.Min(255, grey / 3 + r * 2 / 3) << 16) |
                            (Math.Min(255, grey / 3 + gr * 2 / 3) << 8) |
                            Math.Min(255, grey / 3 + b * 2 / 3);
      } else if (number == NotSent) {
        // The SDK marked it as a player, but it is not sent (dropped by the
        // clean-up, or another "player" such as a ghost on a door).
        colour_pixels_[i] = (Math.Min(255, grey / 2 + 50) << 16) |
                            (grey / 2 << 8) | Math.Min(255, grey / 2 + 50);
      } else {
        colour_pixels_[i] = (grey / 2 << 16) | (grey / 2 << 8) | (grey / 2);
      }
    }
    if (options_.SendDepth) {
      SendDepthImage();
    }
    BitmapData data = depth_bitmap_.LockBits(
        new Rectangle(0, 0, DepthWidth, DepthHeight), ImageLockMode.WriteOnly,
        PixelFormat.Format32bppRgb);
    Marshal.Copy(colour_pixels_, 0, data.Scan0, colour_pixels_.Length);
    depth_bitmap_.UnlockBits(data);
    Invalidate();
  }

  // Sends the camera image to Xenia as XKCL packets, so the games can show
  // the player's face in the player slots: "XKCL", u32 version 1, u32 frame,
  // u16 width, u16 height, u16 first row, u16 rows, then rows * width * 3
  // bytes (B, G, R), little-endian header, top row first, not mirrored.
  const int VideoWidth = 640;
  const int VideoHeight = 480;
  const int VideoRowsPerPacket = 16;
  byte[] video_pixels_;
  uint video_frame_ = 0;
  int video_skipped_ = 0;
  byte[] video_packet_ = new byte[20 + VideoRowsPerPacket * VideoWidth * 3];

  void OnColourFrame(object sender, ColorImageFrameReadyEventArgs e) {
    using (ColorImageFrame frame = e.OpenColorImageFrame()) {
      if (frame == null || frame.Width != VideoWidth ||
          frame.Height != VideoHeight || video_pixels_ == null) {
        return;
      }
      if (video_busy_ || ++video_skipped_ < options_.ColourEvery) {
        return;
      }
      video_skipped_ = 0;
      frame.CopyPixelDataTo(video_pixels_);
    }
    // Crop and send on a worker thread so this SDK callback returns at once.
    byte[] ready = video_pixels_;
    video_pixels_ = video_work_;
    video_work_ = ready;
    video_busy_ = true;
    System.Threading.ThreadPool.QueueUserWorkItem(SendColourWork);
  }

  void SendColourWork(object state) {
    try {
      SendColourImage();
    } catch (Exception ex) {
      // The colour image must never take the bridge down.
      colour_error_ = ex.GetType().Name + ": " + ex.Message;
    } finally {
      video_busy_ = false;
    }
  }

  // Face crop. The games draw the whole colour image in the player slot; on a
  // console the NUI runtime hands them an image already cropped around the
  // player's face. Here a square around the head (found with the SDK's own
  // skeleton-to-colour mapping) is stretched to 640x480; the slot squeezes
  // the 4:3 image into a square again, so the face comes out undistorted.
  const float ColourFocalLength = 531.15f;  // pixels, 640x480 colour camera
  byte[] crop_pixels_ = new byte[VideoWidth * VideoHeight * 4];
  int[] crop_index_ = new int[VideoWidth];
  float[] crop_weight_ = new float[VideoWidth];
  bool crop_valid_ = false;
  float crop_x_, crop_y_, crop_side_;  // centre and side, in colour pixels
  const float DepthFocalLength = 285.63f;  // pixels, 320x240 depth camera
  // Set by the skeleton handler, read by the colour worker (plain values; no
  // SDK calls or skeleton objects are touched from the worker).
  PointF face_depth_point_;
  float face_z_;
  bool face_found_;
  byte[] video_work_;     // the camera image being cropped and sent
  bool video_busy_;
  string colour_error_;   // why the colour image / face crop was switched off

  byte[] ImageToSend() {
    if (!options_.FaceCrop || colour_error_ != null || !face_found_ ||
        face_z_ < 0.5f) {
      crop_valid_ = false;
      return video_work_;
    }
    // The head in depth pixels (320x240) to colour pixels (640x480): both
    // cameras look the same way, so scale by the two focal lengths.
    float scale = ColourFocalLength / DepthFocalLength;
    float head_x = 320f + (face_depth_point_.X - 160f) * scale;
    float head_y = 240f + (face_depth_point_.Y - 120f) * scale +
                   options_.FaceOffset * ColourFocalLength / face_z_;
    float side = options_.FaceSize * ColourFocalLength / face_z_;
    side = Math.Max(64f, Math.Min((float)VideoHeight, side));
    if (!crop_valid_) {
      crop_x_ = head_x;
      crop_y_ = head_y;
      crop_side_ = side;
      crop_valid_ = true;
    } else {
      crop_x_ += (head_x - crop_x_) * 0.4f;
      crop_y_ += (head_y - crop_y_) * 0.4f;
      crop_side_ += (side - crop_side_) * 0.4f;
    }
    float x0 = Math.Max(0f, Math.Min(VideoWidth - crop_side_,
                                     crop_x_ - crop_side_ / 2));
    float y0 = Math.Max(0f, Math.Min(VideoHeight - crop_side_,
                                     crop_y_ - crop_side_ / 2));
    float x_step = crop_side_ / VideoWidth;
    float y_step = crop_side_ / VideoHeight;
    for (int ox = 0; ox < VideoWidth; ++ox) {
      float fx = x0 + (ox + 0.5f) * x_step - 0.5f;
      int ix = Math.Max(0, Math.Min(VideoWidth - 2, (int)fx));
      crop_index_[ox] = ix;
      crop_weight_[ox] = Math.Max(0f, Math.Min(1f, fx - ix));
    }
    for (int oy = 0; oy < VideoHeight; ++oy) {
      float fy = y0 + (oy + 0.5f) * y_step - 0.5f;
      int iy = Math.Max(0, Math.Min(VideoHeight - 2, (int)fy));
      float wy = Math.Max(0f, Math.Min(1f, fy - iy));
      int row_a = iy * VideoWidth * 4;
      int row_b = row_a + VideoWidth * 4;
      int output = oy * VideoWidth * 4;
      for (int ox = 0; ox < VideoWidth; ++ox) {
        int a = row_a + crop_index_[ox] * 4;
        int b = row_b + crop_index_[ox] * 4;
        float wx = crop_weight_[ox];
        for (int ch = 0; ch < 3; ++ch) {
          float top = video_work_[a + ch] +
                      (video_work_[a + 4 + ch] - video_work_[a + ch]) * wx;
          float bottom = video_work_[b + ch] +
                         (video_work_[b + 4 + ch] - video_work_[b + ch]) * wx;
          crop_pixels_[output + ch] = (byte)(top + (bottom - top) * wy + 0.5f);
        }
        output += 4;
      }
    }
    return crop_pixels_;
  }

  void SendColourImage() {
    byte[] pixels;
    try {
      pixels = ImageToSend();
    } catch (Exception ex) {
      colour_error_ = ex.GetType().Name + ": " + ex.Message;
      pixels = video_work_;
    }
    ++video_frame_;
    for (int first = 0; first < VideoHeight; first += VideoRowsPerPacket) {
      int rows = Math.Min(VideoRowsPerPacket, VideoHeight - first);
      byte[] p = video_packet_;
      p[0] = (byte)'X';
      p[1] = (byte)'K';
      p[2] = (byte)'C';
      p[3] = (byte)'L';
      BitConverter.GetBytes((uint)1).CopyTo(p, 4);
      BitConverter.GetBytes(video_frame_).CopyTo(p, 8);
      BitConverter.GetBytes((ushort)VideoWidth).CopyTo(p, 12);
      BitConverter.GetBytes((ushort)VideoHeight).CopyTo(p, 14);
      BitConverter.GetBytes((ushort)first).CopyTo(p, 16);
      BitConverter.GetBytes((ushort)rows).CopyTo(p, 18);
      int o = 20;
      int source = first * VideoWidth * 4;  // the SDK gives B, G, R, unused
      for (int i = 0; i < rows * VideoWidth; ++i) {
        p[o++] = pixels[source];
        p[o++] = pixels[source + 1];
        p[o++] = pixels[source + 2];
        source += 4;
      }
      try {
        udp_.Send(p, o);
      } catch (SocketException) {
      }
    }
  }

  // The players' outlines. The SDK's player mask often swallows whatever
  // touches a player at about the same distance (a bed, a chair, the edge
  // of a door), and the games draw all of it as the player. With the
  // clean-up on, a pixel stays in someone's outline only if it is within a
  // body part's reach of their own skeleton (in metres, in 3D); for people
  // with a position only (no skeleton), within 0.45 m of it sideways.
  // outline_: 0 = nobody, 1-6 = that player's number (sent), NotSent = the
  // SDK marked it as a player but it is not sent (dropped by the clean-up,
  // or someone beyond --players).
  const byte NotSent = 255;
  // Reach around each bone in Bones (torso widest, then hips, limbs).
  static readonly float[] BoneReach = {
      0.33f, 0.33f, 0.22f, 0.24f, 0.20f, 0.20f, 0.20f, 0.24f,
      0.20f, 0.20f, 0.20f, 0.24f, 0.22f, 0.20f, 0.20f,
      0.24f, 0.22f, 0.20f, 0.20f};
  const float PositionReach = 0.45f;
  byte[] outline_ = new byte[DepthWidth * DepthHeight];
  // Per SDK player index (1-6): bones (8 floats each) or a position.
  float[][] bone_data_ = new float[7][];
  int[] bone_count_ = new int[7];
  bool[] clean_index_ = new bool[7];
  float[] position_x_ = new float[7];
  float[] position_z_ = new float[7];
  bool[] position_only_ = new bool[7];
  float[] joint_x_ = new float[Packet.JointCount];
  float[] joint_y_ = new float[Packet.JointCount];
  float[] joint_z_ = new float[Packet.JointCount];
  bool[] joint_ok_ = new bool[Packet.JointCount];

  void BuildOutline() {
    for (int k = 1; k <= 6; ++k) {
      clean_index_[k] = false;
      position_only_[k] = false;
      if (bone_data_[k] == null) {
        bone_data_[k] = new float[19 * 8];
      }
      Skeleton s = k - 1 < skeletons_.Length ? skeletons_[k - 1] : null;
      if (!options_.CleanOutline || number_of_index_[k] == 0 || s == null) {
        continue;
      }
      if (s.TrackingState == SkeletonTrackingState.Tracked) {
        bone_count_[k] = PrepareBones(s, bone_data_[k]);
        // Too little of the body tracked: keep the SDK's outline.
        clean_index_[k] = bone_count_[k] >= 8;
      } else if (s.TrackingState == SkeletonTrackingState.PositionOnly &&
                 s.Position.Z > 0.3f && sensor_ != null) {
        DepthImagePoint d =
            sensor_.CoordinateMapper.MapSkeletonPointToDepthPoint(s.Position,
                                                                  DepthFormat);
        position_x_[k] = (d.X + 0.5f - DepthWidth / 2) * s.Position.Z /
                         DepthFocalLength;
        position_z_[k] = s.Position.Z;
        position_only_[k] = true;
        clean_index_[k] = true;
      }
    }
    for (int v = 0, i = 0; v < DepthHeight; ++v) {
      for (int u = 0; u < DepthWidth; ++u, ++i) {
        int raw = depth_pixels_[i];
        int index = raw & DepthImageFrame.PlayerIndexBitmask;
        if (index == 0) {
          outline_[i] = 0;
          continue;
        }
        int number = index <= 6 ? number_of_index_[index] : 0;
        if (number == 0) {
          outline_[i] = NotSent;
          continue;
        }
        int mm = (raw >> DepthImageFrame.PlayerIndexBitmaskWidth) & 0x1FFF;
        if (!clean_index_[index] || mm <= 0) {
          outline_[i] = (byte)number;
          continue;
        }
        float z = mm * 0.001f;
        float x = (u + 0.5f - DepthWidth / 2) * z / DepthFocalLength;
        byte keep = NotSent;
        if (position_only_[index]) {
          float dx = x - position_x_[index], dz = z - position_z_[index];
          if (dx * dx + dz * dz <= PositionReach * PositionReach) {
            keep = (byte)number;
          }
          outline_[i] = keep;
          continue;
        }
        float y = (DepthHeight / 2 - v - 0.5f) * z / DepthFocalLength;
        float[] bd = bone_data_[index];
        int bones = bone_count_[index];
        for (int b = 0; b < bones * 8; b += 8) {
          // Distance from (x, y, z) to the segment a + t * d, t in 0..1.
          float px = x - bd[b], py = y - bd[b + 1], pz = z - bd[b + 2];
          float t = (px * bd[b + 3] + py * bd[b + 4] + pz * bd[b + 5]) *
                    bd[b + 6];
          t = t < 0f ? 0f : (t > 1f ? 1f : t);
          px -= t * bd[b + 3];
          py -= t * bd[b + 4];
          pz -= t * bd[b + 5];
          if (px * px + py * py + pz * pz <= bd[b + 7]) {
            keep = (byte)number;
            break;
          }
        }
        outline_[i] = keep;
      }
    }
  }

  // Fills bone_data with the skeleton's bones (start, direction, 1 / length
  // squared, reach squared) in the same pinhole space as the depth pixels,
  // so the camera's exact calibration cancels out. Returns the bone count.
  int PrepareBones(Skeleton s, float[] bone_data) {
    KinectSensor sensor = sensor_;
    if (sensor == null) {
      return 0;
    }
    for (int j = 0; j < Packet.JointCount; ++j) {
      Joint joint = s.Joints[(JointType)j];
      joint_ok_[j] = joint.TrackingState != JointTrackingState.NotTracked &&
                     joint.Position.Z > 0.3f;
      if (!joint_ok_[j]) {
        continue;
      }
      DepthImagePoint d = sensor.CoordinateMapper.MapSkeletonPointToDepthPoint(
          joint.Position, DepthFormat);
      float z = joint.Position.Z;
      joint_x_[j] = (d.X + 0.5f - DepthWidth / 2) * z / DepthFocalLength;
      joint_y_[j] = (DepthHeight / 2 - d.Y - 0.5f) * z / DepthFocalLength;
      joint_z_[j] = z;
    }
    int n = 0;
    for (int b = 0; b < Bones.GetLength(0); ++b) {
      int j0 = Bones[b, 0], j1 = Bones[b, 1];
      if (!joint_ok_[j0] || !joint_ok_[j1]) {
        continue;
      }
      float dx = joint_x_[j1] - joint_x_[j0];
      float dy = joint_y_[j1] - joint_y_[j0];
      float dz = joint_z_[j1] - joint_z_[j0];
      float length2 = dx * dx + dy * dy + dz * dz;
      int o = n * 8;
      bone_data[o] = joint_x_[j0];
      bone_data[o + 1] = joint_y_[j0];
      bone_data[o + 2] = joint_z_[j0];
      bone_data[o + 3] = dx;
      bone_data[o + 4] = dy;
      bone_data[o + 5] = dz;
      bone_data[o + 6] = length2 > 1e-6f ? 1f / length2 : 0f;
      bone_data[o + 7] = BoneReach[b] * BoneReach[b];
      ++n;
    }
    return n;
  }

  // Sends the depth image with the player's outline to Xenia as XKDP
  // packets: "XKDP", u32 version 1, u32 frame, u16 width, u16 height,
  // u16 first row, u16 rows, then rows * width u16 (depth mm << 3 | player),
  // little-endian. Player 1 is the player whose skeleton is sent.
  const int DepthRowsPerPacket = 60;
  uint depth_frame_ = 0;
  byte[] depth_packet_ = new byte[20 + DepthRowsPerPacket * DepthWidth * 2];

  void SendDepthImage() {
    ++depth_frame_;
    for (int first = 0; first < DepthHeight; first += DepthRowsPerPacket) {
      int rows = Math.Min(DepthRowsPerPacket, DepthHeight - first);
      byte[] p = depth_packet_;
      p[0] = (byte)'X';
      p[1] = (byte)'K';
      p[2] = (byte)'D';
      p[3] = (byte)'P';
      BitConverter.GetBytes((uint)1).CopyTo(p, 4);
      BitConverter.GetBytes(depth_frame_).CopyTo(p, 8);
      BitConverter.GetBytes((ushort)DepthWidth).CopyTo(p, 12);
      BitConverter.GetBytes((ushort)DepthHeight).CopyTo(p, 14);
      BitConverter.GetBytes((ushort)first).CopyTo(p, 16);
      BitConverter.GetBytes((ushort)rows).CopyTo(p, 18);
      int o = 20;
      for (int i = first * DepthWidth; i < (first + rows) * DepthWidth; ++i) {
        int raw = depth_pixels_[i];
        int mm = (raw >> DepthImageFrame.PlayerIndexBitmaskWidth) & 0x1FFF;
        int number = outline_[i];
        int value = (mm << 3) | (number >= 1 && number <= 6 ? number : 0);
        p[o++] = (byte)(value & 0xFF);
        p[o++] = (byte)(value >> 8);
      }
      try {
        udp_.Send(p, o);
      } catch (SocketException) {
      }
    }
  }

  protected override void OnPaint(PaintEventArgs e) {
    Graphics g = e.Graphics;
    g.InterpolationMode = InterpolationMode.NearestNeighbor;
    g.DrawImage(depth_bitmap_,
                new Rectangle(0, 0, DepthWidth * 2, DepthHeight * 2));
    if (drawn_joints_ != null) {
      g.SmoothingMode = SmoothingMode.AntiAlias;
      using (Pen tracked = new Pen(Color.LimeGreen, 4))
      using (Pen inferred = new Pen(Color.Gold, 2)) {
        for (int b = 0; b < Bones.GetLength(0); ++b) {
          int j0 = Bones[b, 0], j1 = Bones[b, 1];
          if (drawn_states_[j0] == 0 || drawn_states_[j1] == 0) {
            continue;
          }
          bool both = drawn_states_[j0] == 2 && drawn_states_[j1] == 2;
          g.DrawLine(both ? tracked : inferred, Screen(drawn_joints_[j0]),
                     Screen(drawn_joints_[j1]));
        }
      }
      // The player's right hand, to check left/right.
      PointF hand = Screen(drawn_joints_[11]);
      if (options_.Players > 1) {
        DrawNumber(g, primary_number_, Screen(drawn_joints_[3]));
      }
      using (Font font = new Font(FontFamily.GenericSansSerif, 14,
                                  FontStyle.Bold)) {
        g.DrawString("R", font, Brushes.Yellow, hand.X + 6, hand.Y - 10);
      }
    }
    // Everyone else with a full skeleton, in their colour and number.
    g.SmoothingMode = SmoothingMode.AntiAlias;
    foreach (DrawnBody body in drawn_others_) {
      Color colour = PlayerColours[Math.Min(6, Math.Max(1, body.Number))];
      using (Pen pen = new Pen(colour, 3))
      using (Font font = new Font(FontFamily.GenericSansSerif, 12,
                                  FontStyle.Bold))
      using (Brush brush = new SolidBrush(colour)) {
        if (body.States == null) {
          DrawNumber(g, body.Number, Screen(body.Joints[0]));
          continue;
        }
        for (int b = 0; b < Bones.GetLength(0); ++b) {
          int j0 = Bones[b, 0], j1 = Bones[b, 1];
          if (body.States[j0] == 0 || body.States[j1] == 0) {
            continue;
          }
          g.DrawLine(pen, Screen(body.Joints[j0]), Screen(body.Joints[j1]));
        }
        DrawNumber(g, body.Number, Screen(body.Joints[3]));
      }
    }
    string line1 = drawn_joints_ != null
                       ? (options_.Players > 1
                              ? "Main player: P" + primary_number_ + " (ID " +
                                    tracking_id_ + ")"
                              : "Tracking player (ID " + tracking_id_ + ")")
                       : (sensor_ != null ? "Nobody tracked - step in front of "
                                            + "the Kinect, whole body in view"
                                          : "");
    if (options_.Players > 1) {
      line1 += "   People: " + people_ + " (" + full_skeletons_ +
               " full skeletons" +
               (choice_text_.Length > 0 ? "; " + choice_text_ : "") + ")";
    }
    line1 += "   Outline: " + (options_.CleanOutline ? "cleaned (O)"
                                                       : "SDK's (O)");
    string line2 = status_ + "   Skeleton frames/s: " +
                   rate_.ToString("0", CultureInfo.InvariantCulture) +
                   (sensor_ != null ? TiltText() : "") +
                   (options_.FaceCrop && colour_error_ == null
                        ? "   Face crop: " + options_.FaceSize.ToString(
                              "0.00", CultureInfo.InvariantCulture) + " m, " +
                              options_.FaceOffset.ToString(
                                  "0.00", CultureInfo.InvariantCulture) +
                              " m down (Z/X zoom, W/S move)"
                        : "   Whole camera image; face " +
                              options_.FaceOffset.ToString(
                                  "0.00", CultureInfo.InvariantCulture) +
                              " m down (W/S)") +
                   (colour_error_ != null ? "   Colour/face crop off: " +
                                            colour_error_ : "") +
                   "   Q quits";
    using (Font font = new Font(FontFamily.GenericSansSerif, 10)) {
      g.DrawString(line1, font, Brushes.White, 6, DepthHeight * 2 + 6);
      g.DrawString(line2, font, Brushes.Silver, 6, DepthHeight * 2 + 30);
    }
  }

  string SafeTilt() {
    try {
      return sensor_.ElevationAngle.ToString(CultureInfo.InvariantCulture);
    } catch (InvalidOperationException) {
      return "?";
    }
  }

  // A player's number above their head, white on black so it shows on any
  // outline colour.
  static void DrawNumber(Graphics g, int number, PointF head) {
    using (Font font = new Font(FontFamily.GenericSansSerif, 12,
                                FontStyle.Bold)) {
      string text = "P" + number.ToString(CultureInfo.InvariantCulture);
      SizeF size = g.MeasureString(text, font);
      float x = head.X - size.Width / 2, y = head.Y - 40;
      g.FillRectangle(Brushes.Black, x - 2, y, size.Width + 4, size.Height);
      g.DrawString(text, font, Brushes.White, x, y);
    }
  }

  // Depth image point to window point (scaled 2x).
  static PointF Screen(PointF p) { return new PointF(p.X * 2, p.Y * 2); }
}

static class Program {
  [STAThread]
  static int Main(string[] args) {
    Options options = Options.Parse(args);
    if (options == null) {
      return 1;
    }
    Application.EnableVisualStyles();
    Application.Run(new BridgeForm(options));
    return 0;
  }
}

}  // namespace XeniaKinectSdkBridge
