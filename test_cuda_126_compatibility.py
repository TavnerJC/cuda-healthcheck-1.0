# Cell 9: CUDA 12.6 Specific Compatibility Check

from cuda_healthcheck.data import BreakingChangesDatabase

print("=" * 80)
print("🔍 CUDA 12.6 SPECIFIC COMPATIBILITY CHECK")
print("=" * 80)

db = BreakingChangesDatabase()

# Your actual environment
detected_libs = [
    {"name": "pytorch", "version": "2.7.1", "cuda_version": "12.6"},
]

# Score for CUDA 12.6 specifically
print("\n📊 Testing Against CUDA 12.6 (Your Current Runtime):")
score_126 = db.score_compatibility(detected_libs, "12.6")

print(f"\n💯 Compatibility Score: {score_126['compatibility_score']}/100")
print(f"   Critical Issues: {score_126['critical_issues']}")
print(f"   Warning Issues: {score_126['warning_issues']}")
print(f"   Info Issues: {score_126['info_issues']}")
print(f"   Recommendation: {score_126['recommendation']}")

# Show breaking changes specific to 12.6
if score_126['breaking_changes']['CRITICAL']:
    print("\n🔴 CRITICAL Issues for CUDA 12.6:")
    for change in score_126['breaking_changes']['CRITICAL']:
        print(f"   • {change['title']}")
        print(f"     {change['description'][:80]}...")

if score_126['breaking_changes']['WARNING']:
    print("\n⚠️  WARNING Issues for CUDA 12.6:")
    for change in score_126['breaking_changes']['WARNING']:
        print(f"   • {change['title']}")
        print(f"     {change['description']}")

# Check specific 12.4 -> 12.6 transition
print("\n🔄 CUDA 12.4 → 12.6 Transition Analysis:")
changes_124_to_126 = db.get_changes_by_cuda_transition("12.4", "12.6")
print(f"   Found {len(changes_124_to_126)} breaking changes")

for change in changes_124_to_126:
    print(f"\n   📋 {change.title}")
    print(f"      Severity: {change.severity}")
    print(f"      Library: {change.affected_library}")
    print(f"      Description: {change.description}")
    print(f"      Migration: {change.migration_path}")

# Show all CUDA versions we track
print("\n📚 All CUDA Versions Tracked in Database:")
all_changes = db.get_all_changes()
cuda_versions = set()
for change in all_changes:
    if change.cuda_version_from != "Any":
        cuda_versions.add(change.cuda_version_from)
    if change.cuda_version_to != "Any":
        cuda_versions.add(change.cuda_version_to)

sorted_versions = sorted(cuda_versions, key=lambda x: [int(p) if p.isdigit() else 0 for p in x.replace('.x', '.0').split('.')])
print(f"   Versions: {', '.join(sorted_versions)}")

print("\n" + "=" * 80)
print("✅ CUDA 12.6 is tracked in the database!")
print("=" * 80)

