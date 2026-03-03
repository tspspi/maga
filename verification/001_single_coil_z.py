import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from maga import MagneticFieldCalculator, DeviceManager, RectangularGrid
from maga.geometry import CircularCoil

testcases = [
    {
        'center' : (0.0, 0.0, 0.0),
        'normal' : (0.0, 0.0, 1.0),
        'radius' : 10e-2,
        'current' : 1.23,
        'num_elements' : 1024,
        'windings' : 1,
        'name' : "Single 20cm diameter coil, 1.23A",
        'grid_range' : [ (-20e-2, 20e-2), (-20e-2, 20e-2), (-20e-2, 20e-2) ],
        'grid_n' : ( 101, 101, 101 ),
        'expected' : [
            # On axis
            { 'idx' : [50, 50, 50], 'B' : 7.728317926810499e-6 },
            { 'idx' : [50, 50, 45], 'B' : 7.286768476564e-6 },
            { 'idx' : [50, 50, 55], 'B' : 7.286768476564e-6 },
            { 'idx' : [50, 50, 60], 'B' : 6.185830220467e-6 },
            { 'idx' : [50, 50, 65], 'B' : 4.872777904980e-6 },
            { 'idx' : [50, 50, 70], 'B' : 3.679757572745e-6 },

            # Off axis
            { 'idx' : [62, 50, 50], 'B' : 9.43583334142e-6 },
            { 'idx' : [70, 50, 50], 'B' : 1.74434492233e-5 },
            { 'idx' : [60, 50, 55], 'B' : 8.10292686152e-6 },
            { 'idx' : [60, 50, 60], 'B' : 6.50783018640e-6 },
            { 'idx' : [65, 50, 55], 'B' : 9.42425798568e-6 },
            { 'idx' : [55, 50, 60], 'B' : 6.26829749700e-6 },
            { 'idx' : [60, 60, 55], 'B' : 9.12954119524e-6 },
            { 'idx' : [58, 42, 52], 'B' : 9.05538016939e-6 },
            { 'idx' : [54, 56, 66], 'B' : 4.60672133974e-6 }
        ]
    },
    {
        'center' : (0.0, 0.0, 0.0),
        'normal' : (1.0, 0.0, 1.0),
        'radius' : 10e-2,
        'current' : 1.23,
        'num_elements' : 128,
        'windings' : 1,
        'name' : "Single 20cm diameter coil, 1.23A",
        'grid_range' : [ (-20e-2, 20e-2), (-20e-2, 20e-2), (-20e-2, 20e-2) ],
        'grid_n' : ( 101, 101, 101 ),
        'expected' : [
            { 'idx' : [50, 50, 50], 'B' : 7.728317927831e-6 },
            { 'idx' : [55, 50, 55], 'B' : 6.885719808667e-6 },
            { 'idx' : [60, 50, 60], 'B' : 5.095937383929e-6 },
            { 'idx' : [65, 50, 65], 'B' : 3.426039055436e-6 },
            { 'idx' : [70, 50, 70], 'B' : 2.244826131820e-6 }
        ]
    }
]

def biot_savart_ana(
    current = 1,
    windings = 1,
    radius = 1,
    x_offset = 0,
    z_offset = 0
):
    mu0 = 1.25663706127
    return (mu0 * current * radius**2 * windings) / (2 * np.power(R**2 + z**2, 3/2))


report = "# Single coil tests\n\n"

for itc, tc in enumerate(testcases):
    print("========= ========= ========= ========= =========")
    print(tc)

    report = report + f"## Case {itc+1}\n\n"
    report = report + "![](case_sc_" + f"{itc+1}.png)\n\n"
    report = report + f"* Center: {tc['center']}\n"
    report = report + f"* Normal: {tc['normal']}\n"
    report = report + f"* Radius: {tc['radius']} m\n"
    report = report + f"* Current: {tc['current']} A\n"
    report = report + f"* Discrete elements: {tc['num_elements']}\n"
    report = report + f"* Windings: {tc['windings']}\n\n"
    report = report + "| Position | Expected | Result | Deviation | Status |\n"
    report = report + "| --- | --- | --- | --- | --- |\n"

    coil = CircularCoil(
        center = tc['center'],
        normal_vector = tc['normal'],
        radius = tc['radius'],
        current = tc['current'],
        num_elements = tc['num_elements'],
        windings = tc['windings'],
        name = tc['name']
    )
    grid = RectangularGrid(
        x_range = tc['grid_range'][0],
        y_range = tc['grid_range'][1],
        z_range = tc['grid_range'][2],
        nx = tc['grid_n'][0],
        ny = tc['grid_n'][1],
        nz = tc['grid_n'][2],
        name = "Evaluation grid"
    )
    calc = MagneticFieldCalculator()

    A, B, I = coil.get_geometry()
    res = calc.calculate_magnetic_field(A, B, I, grid)

    # field = res.magnetic_field
    field = np.transpose(res.magnetic_field, (2, 1, 0, 3))

    nx, ny, nz = field.shape[:3]
    x = np.linspace(grid.x_range[0], grid.x_range[1], nx)
    y = np.linspace(grid.y_range[0], grid.y_range[1], ny)
    z = np.linspace(grid.z_range[0], grid.z_range[1], nz)

    fig, ax = plt.subplots()
    color = np.log(np.hypot(field[:,50,:,0], field[:,50,:,2]) + 1e-12)

    if 'expected' in tc:
        for expected in tc['expected']:
            # Fetch point from results and calculate absolute field value
            Bz = field[expected['idx'][0], expected['idx'][1], expected['idx'][2], 0]
            By = field[expected['idx'][0], expected['idx'][1], expected['idx'][2], 1]
            Bx = field[expected['idx'][0], expected['idx'][1], expected['idx'][2], 2]

            B = np.sqrt(Bx**2 + By**2 + Bz**2)

            print(f"  Position {expected['idx']}:")
            print(f"    Calculated: {B}")
            print(f"    Expected:   {expected['B']}")

            deviation = (np.abs(B - expected['B']) / expected['B'] * 100)
            if np.abs(B - expected['B']) < (B * 1e-3):
                print(f"    Ok")
                r = "Ok"
            else:
                print(f"    Failed")
                r = "Failed"

            report = report + f"| {expected['idx']} | {expected['B']:.4E} T | {B:.4E} T | {deviation:.5f}% | {r} |\n"


    ax.streamplot(z, x, field[:,50,:,2], field[:,50,:,0], color = color, linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
    plt.tight_layout()
    plt.savefig(f"res/case_sc_{itc+1}.png")
    plt.show()
    plt.close("all")

with open(f"res/sc.md", 'w') as outfile:
    outfile.write(report)

print("Done")
