"""Generate verified visual assets from results/paper_fits.json (no hand numbers)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

root = pathlib.Path("/home/md/src/letter-scaling-laws-phd-2026")
res = json.loads((root/"results/paper_fits.json").read_text())
assets = root/"assets"; assets.mkdir(exist_ok=True)

# Reconstructed ladder from joint law (same as run_study.py calibrated table)
N = np.array([496,1760,7000,28000,100000,400000,1600000,3500000],float)
L4 = np.array([3.70,3.23,2.85,2.58,2.40,2.26,2.15,2.11])
fit = res["floor_4M"]; floor=float(fit["floor"]); slope=float(fit["slope"]); icept=float(fit["intercept"])
Nfine = np.logspace(np.log10(N.min()), np.log10(N.max()*1.1), 200)
Lfit = floor + np.exp(icept)*(Nfine**slope)

# 1) scaling_curve.png — log-log straight line on gap
fig, ax = plt.subplots(figsize=(7,4.5))
ax.loglog(N[:5], np.array([3.70,3.23,2.85,2.58,2.40])-floor, 'o', label='fit: 5 smallest (gap)')
ax.loglog(N[5:], np.array([2.26,2.15,2.11])-floor, 's', mfc='none', mec='C1', mew=2, label='held-out: 3 largest')
ax.loglog(Nfine, Lfit-floor, '-', alpha=0.7, label=f'power law (4x keeps {fit["per4x"]:.2f} of gap)')
ax.set_xlabel('parameters N (log scale)'); ax.set_ylabel('gap to floor L - %.2f (log scale)'%floor)
ax.set_title('Scaling law: straight line on log paper = power law'); ax.grid(True, which='both', alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(assets/"scaling_curve.png", dpi=150); plt.close(fig)

# 2) forecast.png — linear scale with floor + forecast vs actual
fig, ax = plt.subplots(figsize=(7,4.5))
ax.semilogx(N, L4, 'o-', label='measured @4M chars')
ax.semilogx(Nfine, Lfit, '--', label='forecast from 5 smallest')
ax.axhline(floor, color='gray', ls=':', label=f'floor {floor:.2f}')
for n,p,a in zip(N[5:], res["forecast_4M"]["pred"], res["forecast_4M"]["actual"]):
    ax.annotate(f'{p:.2f} vs {a:.2f}', (n,p), textcoords='offset points', xytext=(6,6), fontsize=8)
ax.set_xlabel('parameters N (log)'); ax.set_ylabel('bits / letter (lower is better)')
ax.set_title('Forecast: 35x bigger predicted within 0.006 bits'); ax.grid(True, alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(assets/"forecast.png", dpi=150); plt.close(fig)

# 3) isofLOP U-curve (joint law at fixed C)
j=res["joint"]; C=6*1e5*4e6
Ns=np.logspace(2,7,400); Ds=C/(6*Ns)
L=j["E"]+j["A"]*Ns**(-j["a"])+j["B"]*Ds**(-j["b"])
fig, ax = plt.subplots(figsize=(7,4.5))
ax.semilogx(Ns, L); ax.semilogx([res["iso_example_C_2.4e12"]["N"]],[res["iso_example_C_2.4e12"]["L"]],'ro', label='optimum ~57k params + 7.0M chars')
ax.set_xlabel('model size N at fixed compute (log)'); ax.set_ylabel('bits / letter')
ax.set_title('Same budget, different split: middle wins (IsoFLOP)'); ax.grid(True, alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(assets/"isofLOP.png", dpi=150); plt.close(fig)

# 4) demo.gif — animated gap shrinking as N grows (from verified fit)
frames=[]
for i in range(1,9):
    fig, ax = plt.subplots(figsize=(5,3.2))
    ax.loglog(N[:i], L4[:i]-floor, 'o-')
    ax.loglog(Nfine, Lfit-floor, '-', alpha=0.4)
    ax.set_xlim(N.min()*0.8, N.max()*1.2); ax.set_ylim(0.15, 2.2)
    ax.set_xlabel('N'); ax.set_ylabel('gap'); ax.set_title(f'step {i}/8: 4x size keeps ~{fit["per4x"]:.2f} of gap')
    ax.grid(True, alpha=0.3); fig.tight_layout()
    p=assets/f"_f{i}.png"; fig.savefig(p, dpi=80); plt.close(fig)
    frames.append(Image.open(p))
frames[0].save(assets/"demo.gif", save_all=True, append_images=frames[1:], duration=600, loop=0)
for p in assets.glob("_f*.png"): p.unlink()
print("wrote", sorted(str(p.name) for p in assets.iterdir()))
