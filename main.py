import numpy as np
import scipy.signal

# 1. Définition du signal
Te = 1/1000
t = np.arange(-1, 1, Te)
x = scipy.signal.gausspulse(t, fc=3)

# 2. Paramètres de quantification
q = 5
nb_niveaux = 2**q
# On suppose la dynamique du quantificateur adaptée au signal [-1, 1]
# Pas de quantification (Delta)
Delta = 2 / nb_niveaux 

# 3. Quantification "Mi-montée" (Mid-rise)
# La formule typique est : (floor(x/Delta) + 0.5) * Delta
x_q = (np.floor(x / Delta) + 0.5) * Delta

# 4. Calcul du SQNR en dB
puissance_signal = np.mean(x**2)
puissance_bruit = np.mean((x - x_q)**2)

sqnr = 10 * np.log10(puissance_signal / puissance_bruit)

print(f"La réponse est : {sqnr:.2f}")