import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor

# Metodi di supporto
def significantDigits(x):
    return -int(floor(log10(abs(x))))

def checkInterpol(lines, x, y, dy):
    m, q = lines[0]
    x1 = lines[1]
    x2 = lines[2]
    for xi, yi in zip(x, y):
        if xi != x1 and xi != x2:
            # Calcola il valore della retta nel punto xi
            y_pred = m * xi + q
            # Verifica se y_pred è all'interno della barra di errore [yi - dy, yi + dy]
            if not (yi - dy <= y_pred <= yi + dy):
                return False
    return True

def allLines(x, y, dy):
    lines = []
    for i in range(len(x)):
        x_i, y_i = x[i], y[i] + dy
        for j in range(len(x)):
            if i != j:
                x_j, y_j = x[j], y[j] - dy
                m = (y_j - y_i) / (x_j - x_i)
                q = y_i - m * x_i
                lines.append([(m, q), x_i, x_j])
    return lines

def discrepanza(labels, values, errors):
    # Determinazione dell'intervallo comune alle stime
    min_val = max([v - e for v, e in zip(values, errors)])  # Valore minimo coerente
    max_val = min([v + e for v, e in zip(values, errors)])  # Valore massimo coerente
    extr_inf = []
    extr_sup = []
    for i in range(len(values)):
        extr_inf.append(values[i] - errors[i])
        extr_sup.append(values[i] + errors[i])
    extr_max = max(extr_sup)
    extr_min = min(extr_inf)
    scale = min(errors)/int(str(min(errors))[-1])

    # Creazione del grafico
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(values)), values, yerr=errors, fmt='o', 
                color='red', elinewidth=2, capsize=4, label='Stime con errori')
    plt.scatter(range(len(values)), values, color='red')

    # Aggiunta sfondo per l'intervallo comune
    plt.axhspan(min_val, max_val, color='green', alpha=0.3, label='Intervallo Coerente')

    # Aggiunta delle etichette
    plt.xticks(range(len(values)), labels, fontsize=10)
    plt.yticks(np.arange(extr_min, extr_max+scale, scale))  # Range delle R con passo di 0.1
    plt.ylabel(r'$R_1 \: [\Omega]$', fontsize=12)

    # Aggiunta della legenda e titolo
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def errSist(R_A, R_V, R1):
    r_sist = R_A
    print(f"Errore sistematico commesso dal Circuito 1: {round(r_sist, 4)} " u"\u03A9")
    r_sist = -(R1**2) / (R1 + R_V)
    print(f"Errore sistematico commesso dal Circuito 2: {round(r_sist, 4)} " u"\u03A9")
    

# Metodi principali
def mediaPesata(N, R, dR):
    # Calcolo dei pesi w_i
    w = [1 / (dR[i] ** 2) for i in range(N)]

    # Calcolo di R_best
    R_best = sum(w[i] * R[i] for i in range(N)) / sum(w)

    # Calcolo di delta_R_best
    dR_best = (sum(w)) ** -0.5

    digits = significantDigits(dR_best)
    # Stampa dei risultati
    print(f"R_best = {round(R_best,4)} " u"\u00B1" f" {round(dR_best,4)}" "  ------->  " f"R_best = {round(R_best,digits)} " u"\u00B1" f" {round(dR_best,digits)}")

def metodoGrafico(N, x, dx, y, dy):
    # Calcola tutte le rette possibili
    lines = allLines(x, y, dy)

    right_lines = [line[0] for line in lines if checkInterpol(line, x, y, dy)]
    
    if len(right_lines) == 0:
        print("Nessuna retta interpolante trovata")
        return
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x, y, xerr=dx, yerr=dy,
        fmt='r+', markersize=10, elinewidth=1, capsize=4, label=r"$\delta x, \delta y$"
    )

    # Generazione dei valori x per tracciare le rette
    x_vals = np.linspace(-0.005, max(x)+0.005, 100)
    
    R_max, q_min = max(right_lines, key=lambda x: x[0])
    print(f"R_max = {round(R_max,4)} " u"\u03A9")
    R_min, q_max = min(right_lines, key=lambda x: x[0])
    print(f"R_min = {round(R_min,4)} " u"\u03A9")
    
    R_best = (R_max + R_min) / 2
    dR_best = (R_max - R_min) / 2
    
    digits = significantDigits(dR_best)
    print(f"R_best = {round(R_best,4)} " u"\u00B1" f" {round(dR_best,4)}" "  ------->  " f"R_best = {round(R_best,digits)} " u"\u00B1" f" {round(dR_best,digits)}")
    
    y_vals = R_max * x_vals + q_min
    plt.plot(x_vals, y_vals, 'b-', label=f"y = R_max*x + q_min")
    y_vals = R_min * x_vals + q_max
    plt.plot(x_vals, y_vals, 'g-', label=f"y = R_min*x + q_max")
    
        
    # Etichette, titolo, legenda
    plt.xlabel('I [A]', fontsize=14)
    plt.ylabel('V [V]', fontsize=14)
    plt.axhline(0, color='black', linewidth=0.8)  # Asse orizzontale
    plt.axvline(0, color='black', linewidth=0.8)  # Asse verticale
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    plt.show()

def minimiQuadrati(N, x, dx, y, dy):
    # Calcolo dei termini necessari per la formula
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_squared = sum([i**2 for i in x])
    sum_y_squared = sum([i**2 for i in y])  # Serve solo per il calcolo del coefficiente di correlazione
    sum_x_y = sum([xi * yi for xi, yi in zip(x, y)])

    # Calcolo di A
    numerator_A = (sum_x_squared * sum_y) - (sum_x * sum_x_y)
    denominator = (N * sum_x_squared) - (sum_x**2)
    A = numerator_A / denominator

    # Calcolo di B
    numerator_B = (N * sum_x_y) - (sum_x * sum_y)
    B = numerator_B / denominator

    # Calcolo delle incertezze
    dy_corr = np.sqrt(dy**2 + B**2 * dx**2)         # Solo nel caso dx sia non trascurabile, ovvero dy non sia >> B*dx
    dA = dy_corr * np.sqrt(sum_x_squared / denominator)
    dB = dy_corr * np.sqrt(N / denominator)

    # Calcolo del coefficiente di correlazione
    numerator_r = (N * sum_x_y) - (sum_x * sum_y)
    denominator_r = np.sqrt((N * sum_x_squared - sum_x**2) * (N * sum_y_squared - sum_y**2))
    r = numerator_r / denominator_r

    digits = significantDigits(dA)
    print(f"A = {round(A,4)} " u"\u00B1" f" {round(dA,4)}" "  ------->  " f"A = {round(A,digits)} " u"\u00B1" f" {round(dA,digits)}")
    digits = significantDigits(dB)
    print(f"B = {round(B,4)} " u"\u00B1" f" {round(dB,4)}" "  ------->  " f"B = {round(B,digits)} " u"\u00B1" f" {round(dB,digits)}")
    print(f"dy = {dy} (precedente)")
    print(f"dy' = {round(dy_corr,4)} (corretta)")
    print(u"\u03C1", f"= {round(r,4)}")

    # Calcola i valori della retta interpolante
    x_vals = np.linspace(-0.005, max(x)+0.005, 100)
    y_vals = A + B * x_vals

    # Disegna il grafico con barre d'errore per i dati originali
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x, y, yerr=dy_corr,
        fmt='r+', markersize=10, elinewidth=1, capsize=4, label=r"$\delta y'$"
    )

    # Aggiungi il punto (0, A) con barra d'errore
    plt.errorbar(
        0, A, yerr=2*dA,  # Barra d'errore verticale
        fmt='b+', markersize=10, elinewidth=1, capsize=4, label=r"$A \pm 2 * \delta A$"
    )

    # Aggiungi la retta di interpolazione
    plt.plot(x_vals, y_vals, label=r"$y = A + Bx$", color="orange", linestyle="-", linewidth=1.5)

    # Etichette, titolo, legenda
    plt.xlabel('i [A]', fontsize=14)
    plt.ylabel('V [V]', fontsize=14)
    plt.axhline(0, color='black', linewidth=0.8)  # Asse orizzontale
    plt.axvline(0, color='black', linewidth=0.8)  # Asse verticale
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    plt.show()


# Definizione delle costanti
N = 10
dV = 0.04
dI = 0.001
labels = ['Minimi Quadrati', 'Metodo Grafico', 'Media Pesata']

# Inizializzazione delle liste V, I, R, dR
# Dati Circuito 1
I1 = [0.0427, 0.0383, 0.0340, 0.0290, 0.0246, 0.0210, 0.0170, 0.0134, 0.0089, 0.0057]
V1 = [2.00, 1.80, 1.60, 1.40, 1.20, 1.00, 0.80, 0.60, 0.40, 0.26]
R1 = [46.8384, 46.9974, 47.0588, 48.2759, 48.7805, 47.6190, 47.0588, 44.7761, 44.9438, 46.3158]
dR1 = [1.4425, 1.6114, 1.8165, 2.1619, 2.5644, 2.9614, 3.6331, 4.4807, 6.7602, 10.7364]
stime1 = [48, 48, 47.7]
dstime1 = [2, 1, 0.7]
# Dati Circuito 2
I2 = [0.0500, 0.0450, 0.0400, 0.0350, 0.0300, 0.0250, 0.0200, 0.0150, 0.0100, 0.0057]
V2 = [1.6560, 1.5200, 1.3600, 1.1920, 1.0160, 0.8400, 0.6760, 0.5120, 0.3480, 0.1920]
R2 = [33.1200, 33.7778, 34.0000, 34.0571, 33.8667, 33.6000, 33.8000, 34.1333, 34.8000, 33.6842]
dR2 = [1.0386, 1.1634, 1.3124, 1.5010, 1.7470, 2.0896, 2.6184, 3.5056, 5.3019, 9.1743]
stime2 = [33, 33, 33.7]
dstime2 = [1, 1, 0.5]


if __name__ == "__main__":
    domanda1 = """
Quale circuito vuoi analizzare?
1 --> Circuito 1
2 --> Circuito 2
3 --> Discrepanza tra i 2 circuiti
Inserisci il numero: """

    domanda2 = """
Quale metodo tra i seguenti ti interessa?
1 --> Media pesata dei valori di R
2 --> Metodo grafico
3 --> Interpolazione lineare tramite minimi quadrati
4 --> Discrepanza tra i 3 metodi
Inserisci il numero: """

    circuito = 0
    metodo = 0

    while circuito not in [1, 2, 3]:
        circuito = int(input(domanda1))
        if circuito not in [1, 2, 3]:
            print("\nInserisci un valore tra 1 e 3!")
            
    if circuito == 1:
        I = I1
        V = V1
        R = R1
        dR = dR1
        stime = stime1
        dstime = dstime1
    elif circuito == 2:
        I = I2
        V = V2
        R = R2
        dR = dR2
        stime = stime2
        dstime = dstime2
    elif circuito == 3:
        print()
        errSist(R_A = 0.5/0.05, R_V = 20000/2, R1 = 33)  #0.05A = I_FS,     2V = V_FS,      R1 = stima di R1 tramite interpolazione dal circuito 2
        exit()
        
        
    while metodo not in [1, 2, 3, 4]:
        metodo = int(input(domanda2))
        if metodo not in [1, 2, 3, 4]:
            print("\nInserisci un valore tra 1 e 4!")

    print()
    if metodo == 1:
        # Esegui la media pesata dei valori di R
        mediaPesata(N, R, dR)
    elif metodo == 2:
        # Esegui il metodo grafico
        metodoGrafico(N, I, dI, V, dV)
    elif metodo == 3:
        # Esegui l'interpolazione lineare tramite minimi quadrati
        minimiQuadrati(N, I, dI, V, dV)
    elif metodo == 4:
        # Visualizza la discrepanza tra i 3 metodi
        discrepanza(labels, stime, dstime)