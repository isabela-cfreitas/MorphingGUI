import numpy as np
from .morph_core import indices_delaunay  # os alunos podem reutilizar

# ------------------------- Funções a implementar pelos estudantes -------------------------

def pontos_medios(pA, pB):
    """
    Retorna os pontos médios (N,2) entre pA e pB.
    """
    return (pA + pB)/2.0

def indices_pontos_medios(pA, pB):
    """
    Calcula a triangulação de Delaunay nos pontos médios e retorna (M,3) int.
    Dica: use pontos_medios + indices_delaunay().
    """
    pt_medio = pontos_medios(pA,pB)
    return indices_delaunay(pt_medio)

# Interpoladoras
def linear(t, a=1.0, b=0.0):
    """
    Interpolação linear: a*t + b (espera-se mapear t em [0,1]).
    """
    return t*a + b #acho que isso é aquilo de deformação dependente do tempo

def sigmoide(t, k):
    """
    Sigmoide centrada em 0.5, normalizada para [0,1].
    k controla a "inclinação": maior k => transição mais rápida no meio.
    """
    return 1 / (1 + np.e ** (-k*(t-0.5)))

def dummy(t):
    """
    Função 'dummy' que pode ser usada como exemplo de função constante.
    """
    return np.ones_like(t)

# Geometria / warping por triângulos
def _det3(a, b, c):
    """
    Determinante 2D para área assinada (auxiliar das baricêntricas).
    """
    return (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))

def _transf_baricentrica(pt, tri):
    """
    pt: (x,y)
    tri: (3,2) com vértices v1,v2,v3
    Retorna (w1,w2,w3); espera-se w1+w2+w3=1 quando pt está no plano do tri.
    """
    area_total = _det3(tri[0], tri[1], tri[2])

    w1 = _det3(pt, tri[1], tri[2]) / area_total
    w2 = _det3(tri[0], pt, tri[2]) / area_total
    w3 = _det3(tri[0], tri[1], pt) / area_total

    return w1, w2, w3

def _check_bari(w1, w2, w3, eps=1e-4):
    """
    Testa inclusão de ponto no triângulo usando baricêntricas (com tolerância).
    """
    return (w1>=-eps) and (w2>=-eps) and (w3>=-eps) and abs(w1 + w2 + w3 - 1) <= eps

def _tri_bbox(tri, W, H):
    """
    Retorna bounding box inteiro (xmin,xmax,ymin,ymax), recortado ao domínio [0..W-1],[0..H-1].
    """
    xmin = max(int(np.floor(np.min(tri[:,0]))), 0)
    xmax = min(int(np.ceil(np.max(tri[:,0]))), W-1)
    ymin = max(int(np.floor(np.min(tri[:,1]))), 0)
    ymax = min(int(np.ceil(np.max(tri[:,1]))), H-1)
    return xmin, xmax, ymin, ymax

def _amostra_bilinear(img_float, x, y):
    """
    Amostragem bilinear em (x,y) com clamp nas bordas.
    img_float: (H,W,3) float32 [0,1] — retorna vetor (3,).
    """
    #img_float[y][x][canal]
    pixel = []
    h, w = img_float.shape[0:2]
    for i in range(3):
        x_left = int(np.clip(np.floor(x), 0, w-1)) #a1 = y_up, x_left
        x_right = int(np.clip(np.ceil(x), 0, w-1)) #a2 = y_up, x_right
        y_down = int(np.clip(np.ceil(y), 0, h-1))  #a3 = y_down, x_left
        y_up = int(np.clip(np.floor(y), 0, h-1))   #a4 = y_down, x_right
        left = x_right - x
        right = 1 - left
        up = y_down - y
        down = 1 - up
        p = ( up * left * img_float[y_up][x_left][i]
            + up * right * img_float[y_up][x_right][i]
            + down * left * img_float[y_down][x_left][i]
            + down * right * img_float[y_down][x_right][i] )
        pixel.append(p)
    return pixel

def gera_frame(A, B, pA, pB, triangles, alfa, beta):
    """
    Gera um frame intermediário por morphing com warping por triângulos.
    - A,B: imagens (H,W,3) float32 em [0,1]
    - pA,pB: (N,2) pontos correspondentes
    - triangles: (M,3) índices de triângulos
    - alfa: controla geometria (0=A, 1=B)
    - beta:  controla mistura de cores (0=A, 1=B)
    Retorna (H,W,3) float32 em [0,1].
    """

    H, W = A.shape[0:2]
    n_pixels = H * W
    count = 0
    C = np.zeros_like(A)
    N = np.zeros((H, W))
    pT = (1-alfa) * pA + alfa * pB #pts itermediarios entre imgs

    for t in triangles:
        a,b,c = t
        
        tri_t = np.array([pT[a], pT[b], pT[c]]) #coordenadas do triangulo na imagem intermediaria
        tri_a = np.array([pA[a], pA[b], pA[c]]) #coordenadas do triangulo na imagem A
        tri_b = np.array([pB[a], pB[b], pB[c]]) #coordenadas do triangulo na imagem B

        xmin, xmax, ymin, ymax = _tri_bbox(tri_t, W, H)

        for x in range(xmin, xmax+1):
            for y in range(ymin, ymax+1):
                w1, w2, w3 = _transf_baricentrica([x, y], tri_t)
                if _check_bari(w1, w2, w3):
                    count += 1
                    print(f"x: {x}, y: {y}, alfa: {alfa}, progresso: {100*count/n_pixels} %")
                    xA = w1*tri_a[0,0] + w2*tri_a[1,0] + w3*tri_a[2,0]
                    yA = w1*tri_a[0,1] + w2*tri_a[1,1] + w3*tri_a[2,1]
                    
                    xB = w1*tri_b[0,0] + w2*tri_b[1,0] + w3*tri_b[2,0]
                    yB = w1*tri_b[0,1] + w2*tri_b[1,1] + w3*tri_b[2,1]

                    corA = _amostra_bilinear(A,xA,yA)
                    corB = _amostra_bilinear(B,xB,yB)
                    N[y][x] += 1
                    for i in range(3):
                        C[y][x][i] = ((1-beta) * corA[i] + beta * corB[i] + C[y][x][i] * (N[y][x]-1))  / N[y][x]
    
    # #v(p) = α(t) · v(p1) + (1 − α(t)) · v(p2)
    # I = (1-beta) * A * (X_A) + beta * B * (X_B)
    print(N)
    return C
