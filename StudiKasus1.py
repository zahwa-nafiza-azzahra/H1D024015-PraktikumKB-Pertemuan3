import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

barang_terjual = ctrl.Antecedent(np.arange(0, 101, 1), 'barang_terjual')
permintaan = ctrl.Antecedent(np.arange(0, 301, 1), 'permintaan')
harga_item = ctrl.Antecedent(np.arange(0, 100001, 1), 'harga_item')
profit = ctrl.Antecedent(np.arange(0, 4000001, 1), 'profit')
stok_makanan = ctrl.Consequent(np.arange(0, 1001, 1), 'stok_makanan')

barang_terjual['rendah'] = fuzz.trapmf(barang_terjual.universe, [0, 0, 20, 40])
barang_terjual['sedang'] = fuzz.trimf(barang_terjual.universe, [30, 50, 70])
barang_terjual['tinggi'] = fuzz.trapmf(barang_terjual.universe, [60, 80, 100, 100])

permintaan['rendah'] = fuzz.trapmf(permintaan.universe, [0, 0, 50, 100])
permintaan['sedang'] = fuzz.trimf(permintaan.universe, [50, 150, 250])
permintaan['tinggi'] = fuzz.trapmf(permintaan.universe, [150, 250, 300, 300])

harga_item['murah'] = fuzz.trapmf(harga_item.universe, [0, 0, 20000, 40000])
harga_item['sedang'] = fuzz.trimf(harga_item.universe, [30000, 50000, 70000])
harga_item['mahal'] = fuzz.trapmf(harga_item.universe, [60000, 80000, 100000, 100000])

profit['rendah'] = fuzz.trapmf(profit.universe, [0, 0, 1000000, 2000000])
profit['sedang'] = fuzz.trimf(profit.universe, [1000000, 2500000, 3500000])
profit['tinggi'] = fuzz.trapmf(profit.universe, [3000000, 3500000, 4000000, 4000000])

stok_makanan['sedang'] = fuzz.trapmf(stok_makanan.universe, [0, 0, 400, 600])
stok_makanan['banyak'] = fuzz.trapmf(stok_makanan.universe, [500, 800, 1000, 1000])

rule1 = ctrl.Rule(barang_terjual['tinggi'] & permintaan['tinggi'] & harga_item['murah'] & profit['tinggi'], stok_makanan['banyak'])
rule2 = ctrl.Rule(barang_terjual['tinggi'] & permintaan['tinggi'] & harga_item['murah'] & profit['sedang'], stok_makanan['sedang'])
rule3 = ctrl.Rule(barang_terjual['tinggi'] & permintaan['sedang'] & harga_item['murah'] & profit['sedang'], stok_makanan['sedang'])
rule4 = ctrl.Rule(barang_terjual['sedang'] & permintaan['tinggi'] & harga_item['murah'] & profit['sedang'], stok_makanan['sedang'])
rule5 = ctrl.Rule(barang_terjual['sedang'] & permintaan['tinggi'] & harga_item['murah'] & profit['tinggi'], stok_makanan['banyak'])
rule6 = ctrl.Rule(barang_terjual['rendah'] & permintaan['rendah'] & harga_item['sedang'] & profit['sedang'], stok_makanan['sedang'])

stok_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
stok_sim = ctrl.ControlSystemSimulation(stok_ctrl)

stok_sim.input['barang_terjual'] = 80
stok_sim.input['permintaan'] = 255
stok_sim.input['harga_item'] = 25000
stok_sim.input['profit'] = 3500000

stok_sim.compute()

print(f"Hasil Perhitungan Toko Hewan:")
print(f"Jumlah Stok Makanan Optimal: {stok_sim.output['stok_makanan']:.2f} unit")

stok_makanan.view(sim=stok_sim)
plt.show()