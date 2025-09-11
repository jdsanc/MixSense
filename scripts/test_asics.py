# scripts/test_asics.py
import numpy as np
from app.tools_asics import asics_quantify

ppm = np.arange(0, 10, 0.002).tolist()

def g(center, area, width=0.01):
    x = np.array(ppm)
    return (area * np.exp(-0.5*((x-center)/width)**2)).tolist()

# references (toy peaks)
acetone = {"name":"acetone", "ppm": ppm, "intensity": g(2.10, 6)}
formald = {"name":"formaldehyde", "ppm": ppm, "intensity": g(9.75, 2)}
product = {"name":"diacetone_alcohol", "ppm": ppm,
           "intensity": (np.array(g(3.60,1))+np.array(g(2.15,3))+np.array(g(1.05,6))).tolist()}

# mixture = 0.25 A + 0.05 F + 0.70 P
y = (0.25*np.array(acetone["intensity"]) +
     0.05*np.array(formald["intensity"]) +
     0.70*np.array(product["intensity"]))
y = (y / (y.max()+1e-12)).tolist()

out = asics_quantify(
    crude_ppm=ppm,
    crude_intensity=y,
    refs=[acetone, formald, product],
    nb_protons={"acetone":6, "formaldehyde":2, "diacetone_alcohol":12},
    exclusion_ranges=[[4.5,5.1]],  # water (if present)
    max_shift=0.02,
    quant_method="FWER"
)
print(out["components"])
