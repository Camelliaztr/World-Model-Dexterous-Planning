from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = Path("reports/figures_compare")
out_dir.mkdir(parents=True, exist_ok=True)

methods = ["BC", "WV n=0.08", "WV n=0.01"]

success_rates = [1.0, 0.84, 1.0]
avg_returns = [4435.032928368929, 2993.682969332866, 4414.037310720415]

plt.figure(figsize=(7, 5))
plt.bar(methods, success_rates)
plt.ylabel("Success Rate")
plt.title("Final Method Success Rate Comparison")
plt.ylim(0, 1.05)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "final_method_success_rate_comparison.png", dpi=220)
plt.close()

plt.figure(figsize=(7, 5))
plt.bar(methods, avg_returns)
plt.ylabel("Average Return")
plt.title("Final Method Average Return Comparison")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "final_method_avg_return_comparison.png", dpi=220)
plt.close()

print("Saved final comparison figures to:", out_dir)
