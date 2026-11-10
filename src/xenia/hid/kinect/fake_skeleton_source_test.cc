// Standalone sanity test (no Xenia deps). Build from the repo's src/ dir:
//   g++ -std=c++17 -I. xenia/hid/kinect/fake_skeleton_source.cc
//       xenia/hid/kinect/fake_skeleton_source_test.cc -o skel_test
//   then run ./skel_test
#include <cmath>
#include <cstdio>

#include "xenia/hid/kinect/fake_skeleton_source.h"

using namespace xe::hid::kinect;

static int g_failures = 0;
#define CHECK(cond)                                              \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                              \
    }                                                            \
  } while (0)

static bool Near(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) <= eps;
}
static const Joint& J(const SkeletonFrame& f, JointId id) {
  return f.skeletons[0].joints[static_cast<size_t>(id)];
}

int main() {
  FakeSkeletonSource src;
  SkeletonFrame f = src.GenerateFrame(1, 0);

  // Structure.
  CHECK(kJointCount == 20);
  CHECK(f.skeletons[0].state == TrackingState::kTracked);
  CHECK(f.skeletons[0].tracking_id == 1);
  for (size_t i = 1; i < kMaxSkeletons; ++i)
    CHECK(f.skeletons[i].state == TrackingState::kNotTracked);
  for (const Joint& j : f.skeletons[0].joints)
    CHECK(j.state == TrackingState::kTracked && j.position.w == 1.0f);

  // Left/right symmetry about the body axis at rest.
  CHECK(Near(J(f, JointId::kShoulderLeft).position.x,
             -J(f, JointId::kShoulderRight).position.x));
  CHECK(Near(J(f, JointId::kKneeLeft).position.y,
             J(f, JointId::kKneeRight).position.y));

  // User-right has larger x (assumed sensor-space convention).
  CHECK(J(f, JointId::kHandRight).position.x >
        J(f, JointId::kHandLeft).position.x);

  // Feet lie on the floor plane; the head is above the hips; z == distance.
  const Vec4& p = f.floor_clip_plane;
  const Vec4 foot = J(f, JointId::kFootLeft).position;
  CHECK(Near(p.x * foot.x + p.y * foot.y + p.z * foot.z + p.w, 0.0f));
  CHECK(J(f, JointId::kHead).position.y > J(f, JointId::kHipCenter).position.y);
  CHECK(Near(J(f, JointId::kHipCenter).position.z, 2.5f));

  // Manual hand control: raised toward the sensor, offset from the shoulder.
  src.SetRightHandOffset(0.10f, 0.20f);
  SkeletonFrame m = src.GenerateFrame(2, 33333);
  const Vec4 sh = J(m, JointId::kShoulderRight).position;
  const Vec4 hand = J(m, JointId::kHandRight).position;
  CHECK(Near(hand.x - sh.x, 0.10f) && Near(hand.y - sh.y, 0.20f));
  CHECK(hand.z < sh.z);  // Closer to the sensor.
  src.ClearRightHand();
  SkeletonFrame r = src.GenerateFrame(3, 66666);
  CHECK(Near(J(r, JointId::kHandRight).position.x,
             J(f, JointId::kHandRight).position.x));

  // Scripted circle returns to its start after one period, stays on radius.
  FakeSkeletonConfig cc;
  cc.hand_motion = FakeHandMotion::kCircle;
  cc.motion_period_s = 4.0f;
  cc.motion_radius_m = 0.25f;
  FakeSkeletonSource circ(cc);
  SkeletonFrame c0 = circ.GenerateFrame(1, 0);
  SkeletonFrame c1 = circ.GenerateFrame(2, 4000000);
  SkeletonFrame ch = circ.GenerateFrame(3, 1000000);
  CHECK(Near(J(c0, JointId::kHandRight).position.x,
             J(c1, JointId::kHandRight).position.x, 1e-3f));
  const Vec4 csh = J(ch, JointId::kShoulderRight).position;
  const Vec4 chd = J(ch, JointId::kHandRight).position;
  CHECK(Near(std::hypot(chd.x - csh.x, chd.y - csh.y), 0.25f, 1e-3f));

  // Polling cadence with an injected clock: 30 Hz, newest-frame semantics.
  uint64_t now = 1000;
  FakeSkeletonSource timed(FakeSkeletonConfig(), [&] { return now; });
  SkeletonFrame out;
  CHECK(!timed.TryGetLatestFrame(&out));  // Not started.
  timed.Start();
  CHECK(timed.TryGetLatestFrame(&out) && out.frame_number == 1);
  CHECK(!timed.TryGetLatestFrame(&out));  // Same frame period.
  now += 10000;
  CHECK(!timed.TryGetLatestFrame(&out));  // 10 ms < 33 ms.
  now += 24000;                           // 34 ms total.
  CHECK(timed.TryGetLatestFrame(&out) && out.frame_number == 2);
  now += 500000;                          // Skips ahead; no backlog replay.
  CHECK(timed.TryGetLatestFrame(&out) && out.frame_number >= 16);
  uint64_t last = out.frame_number;
  CHECK(!timed.TryGetLatestFrame(&out));
  timed.Stop();
  now += 100000;
  CHECK(!timed.TryGetLatestFrame(&out));
  (void)last;

  if (g_failures == 0) std::printf("ALL PASSED\n");
  return g_failures ? 1 : 0;
}
