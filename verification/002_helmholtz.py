import os
import numpy as np
import matplotlib.pyplot as plt

from maga import MagneticFieldCalculator, DeviceManager, RectangularGrid
from maga.geometry.coil_pairs import HelmholtzCoils

# Ensure result directory exists for plots/reports
os.makedirs("res", exist_ok=True)

# Shared grid definition
GRID_RANGE = [(-20e-2, 20e-2), (-20e-2, 20e-2), (-20e-2, 20e-2)]
GRID_SIZE = (101, 101, 101)

testcases = [
    {
        'name': "Helmholtz pair (axis z)",
        'center': (0.0, 0.0, 0.0),
        'normal': (0.0, 0.0, 1.0),
        'radius': 10e-2,
        'current': 2.34,
        'windings': 2,
        'num_elements': 256,
        'grid_range': GRID_RANGE,
        'grid_n': GRID_SIZE,
        'expected': [
            { 'idx': [50, 50, 50], 'B': 4.208145016482636e-05 },
            { 'idx': [50, 50, 45], 'B': 4.200723959390455e-05 },
            { 'idx': [50, 50, 55], 'B': 4.200723959390455e-05 },
            { 'idx': [50, 50, 60], 'B': 4.104525361431799e-05 },
            { 'idx': [50, 50, 65], 'B': 3.791997171392152e-05 },
            { 'idx': [50, 50, 35], 'B': 3.791997171392152e-05 },
            { 'idx': [50, 50, 70], 'B': 3.250456741641173e-05 },
            { 'idx': [62, 50, 50], 'B': 4.090650533558017e-05 },
            { 'idx': [70, 50, 50], 'B': 3.085569623454222e-05 },
            { 'idx': [60, 50, 55], 'B': 4.261942485008114e-05 },
            { 'idx': [60, 50, 60], 'B': 4.3786436542103495e-05 },
            { 'idx': [65, 50, 55], 'B': 4.2466616852065896e-05 },
            { 'idx': [48, 50, 50], 'B': 4.208070118899503e-05 },
            { 'idx': [52, 50, 50], 'B': 4.208070118899503e-05 }
        ]
    },
    {
        'name': "Helmholtz pair (axis 1,0,1)",
        'center': (0.0, 0.0, 0.0),
        'normal': (1.0, 0.0, 1.0),
        'radius': 10e-2,
        'current': 2.34,
        'windings': 2,
        'num_elements': 256,
        'grid_range': GRID_RANGE,
        'grid_n': GRID_SIZE,
        'expected': [
            { 'idx': [50, 50, 50], 'B': 4.208145016482636e-05 },
            { 'idx': [55, 50, 55], 'B': 4.179764010321709e-05 },
            { 'idx': [60, 50, 60], 'B': 3.863751578683737e-05 },
            { 'idx': [65, 50, 65], 'B': 3.097387209660307e-05 },
            { 'idx': [70, 50, 70], 'B': 2.1974752545690724e-05 },
            { 'idx': [60, 60, 55], 'B': 4.4177694107291996e-05 },
            { 'idx': [58, 42, 52], 'B': 4.3045201515019134e-05 },
            { 'idx': [54, 56, 66], 'B': 4.2280937381096215e-05 },
            { 'idx': [45, 45, 45], 'B': 4.2171029060027855e-05 },
            { 'idx': [55, 45, 55], 'B': 4.2171029060027855e-05 }
        ]
    }
]

report = "# Helmholtz pair tests\n\n"

for itc, tc in enumerate(testcases):
    print("========= ========= ========= ========= =========")
    print(tc)

    report += f"## Case {itc + 1}\n\n"
    report += f"![](case_hh_{itc + 1}.png)\n\n"
    report += f"* Center: {tc['center']}\n"
    report += f"* Normal: {tc['normal']}\n"
    # report += f"* Radius: {tc['radius']} m\n"
    report += f"* Current: {tc['current']} A\n"
    report += f"* Windings: {tc['windings']}\n"
    report += f"* Discrete elements per coil: {tc['num_elements']}\n\n"
    report += "| Position | Expected | Result | Deviation | Status |\n"
    report += "| --- | --- | --- | --- | --- |\n"

    helmholtz = HelmholtzCoils(
        center=tc['center'],
        axis=tc['normal'],
        radius=tc['radius'],
        current=tc['current'], # (leave separation 
        windings=tc['windings'],
        num_elements_per_coil=tc['num_elements'],
        separation=tc['radius'],  # classic Helmholtz spacing
        name=tc['name']
    )

    geometry_A, geometry_B, geometry_I = helmholtz.get_geometry()

    grid = RectangularGrid(
        x_range=tc['grid_range'][0],
        y_range=tc['grid_range'][1],
        z_range=tc['grid_range'][2],
        nx=tc['grid_n'][0],
        ny=tc['grid_n'][1],
        nz=tc['grid_n'][2],
        name="Evaluation grid"
    )

    calc = MagneticFieldCalculator()
    res = calc.calculate_magnetic_field(geometry_A, geometry_B, geometry_I, grid)

    field = np.transpose(res.magnetic_field, (2, 1, 0, 3))

    nx, ny, nz = field.shape[:3]
    x = np.linspace(grid.x_range[0], grid.x_range[1], nx)
    y = np.linspace(grid.y_range[0], grid.y_range[1], ny)
    z = np.linspace(grid.z_range[0], grid.z_range[1], nz)

    mid_y = ny // 2
    color = np.log(np.hypot(field[:, mid_y, :, 0], field[:, mid_y, :, 2]) + 1e-12)

    if 'expected' in tc:
        for expected in tc['expected']:
            idx_x, idx_y, idx_z = expected['idx']
            Bx = field[idx_x, idx_y, idx_z, 0]
            By = field[idx_x, idx_y, idx_z, 1]
            Bz = field[idx_x, idx_y, idx_z, 2]
            B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

            print(f"  Position {expected['idx']}:")
            print(f"    Calculated: {B_mag}")
            print(f"    Expected:   {expected['B']}")

            deviation = (np.abs(B_mag - expected['B']) / expected['B'] * 100)
            if np.abs(B_mag - expected['B']) < (B_mag * 1e-3):
                print("    Ok")
                status = "Ok"
            else:
                print("    Failed")
                status = "Failed"

            report += f"| {expected['idx']} | {expected['B']:.4E} T | {B_mag:.4E} T | {deviation:.4f}% | {status} |\n"

    fig, ax = plt.subplots()
    ax.streamplot(z, x, field[:, mid_y, :, 2], field[:, mid_y, :, 0], color=color,
                  linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
    plt.tight_layout()
    plt.savefig(f"res/case_hh_{itc + 1}.png")
    plt.show()
    plt.close("all")

with open("res/helmholtz.md", "w") as outfile:
    outfile.write(report)

print("Done")
