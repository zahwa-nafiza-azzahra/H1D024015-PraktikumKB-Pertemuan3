import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

informasi = ctrl.Antecedent(np.arange(0, 101, 1), 'informasi')
persyaratan = ctrl.Antecedent(np.arange(0, 101, 1), 'persyaratan')
petugas = ctrl.Antecedent(np.arange(0, 101, 1), 'petugas')
sarpras = ctrl.Antecedent(np.arange(0, 101, 1), 'sarpras')
kepuasan = ctrl.Consequent(np.arange(0, 401, 1), 'kepuasan')

for var in [informasi, persyaratan, petugas, sarpras]:
    var['tidak_memuaskan'] = fuzz.trapmf(var.universe, [0, 0, 60, 75])
    var['cukup_memuaskan'] = fuzz.trimf(var.universe, [60, 75, 90])
    var['memuaskan'] = fuzz.trapmf(var.universe, [75, 90, 100, 100])

kepuasan['tidak_memuaskan'] = fuzz.trapmf(kepuasan.universe, [0, 0, 50, 100])
kepuasan['kurang_memuaskan'] = fuzz.trimf(kepuasan.universe, [75, 125, 175])
kepuasan['cukup_memuaskan'] = fuzz.trimf(kepuasan.universe, [150, 225, 300])
kepuasan['memuaskan'] = fuzz.trimf(kepuasan.universe, [250, 300, 350])
kepuasan['sangat_memuaskan'] = fuzz.trapmf(kepuasan.universe, [325, 375, 400, 400])

rules = [
    ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['tidak_memuaskan'] & sarpras['tidak_memuaskan'], kepuasan['tidak_memuaskan']),
    ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['tidak_memuaskan'] & sarpras['cukup_memuaskan'], kepuasan['tidak_memuaskan']),
    ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['tidak_memuaskan'] & sarpras['memuaskan'], kepuasan['tidak_memuaskan']),
    ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['cukup_memuaskan'] & sarpras['tidak_memuaskan'], kepuasan['tidak_memuaskan']),
    ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['cukup_memuaskan'] & sarpras['cukup_memuaskan'], kepuasan['tidak_memuaskan']),
    ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['cukup_memuaskan'] & sarpras['memuaskan'], kepuasan['cukup_memuaskan']),
    ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['memuaskan'] & sarpras['tidak_memuaskan'], kepuasan['tidak_memuaskan']),
    ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['memuaskan'] & sarpras['cukup_memuaskan'], kepuasan['cukup_memuaskan']),
    ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['memuaskan'] & sarpras['memuaskan'], kepuasan['cukup_memuaskan']),
    ctrl.Rule(informasi['cukup_memuaskan'] & persyaratan['cukup_memuaskan'] & petugas['cukup_memuaskan'] & sarpras['memuaskan'], kepuasan['memuaskan']),
    ctrl.Rule(informasi['cukup_memuaskan'] & persyaratan['cukup_memuaskan'] & petugas['memuaskan'] & sarpras['memuaskan'], kepuasan['memuaskan']),
    ctrl.Rule(informasi['cukup_memuaskan'] & persyaratan['memuaskan'] & petugas['memuaskan'] & sarpras['memuaskan'], kepuasan['sangat_memuaskan']),
    ctrl.Rule(informasi['memuaskan'] & persyaratan['memuaskan'] & petugas['memuaskan'] & sarpras['memuaskan'], kepuasan['sangat_memuaskan']),
    
    ctrl.Rule(informasi['memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['tidak_memuaskan'] & sarpras['memuaskan'], kepuasan['cukup_memuaskan'])
]

kepuasan_ctrl = ctrl.ControlSystem(rules)
kepuasan_sim = ctrl.ControlSystemSimulation(kepuasan_ctrl)

kepuasan_sim.input['informasi'] = 80
kepuasan_sim.input['persyaratan'] = 60
kepuasan_sim.input['petugas'] = 50
kepuasan_sim.input['sarpras'] = 90

kepuasan_sim.compute()

print("--- Hasil Perhitungan Pelayanan Masyarakat ---")
print(f"Nilai Tingkat Kepuasan: {kepuasan_sim.output['kepuasan']:.2f}")

kepuasan.view(sim=kepuasan_sim)
plt.show()