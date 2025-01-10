import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Caminho da imagem
img_path = 'OCT_6_20X_TRANS_25C_2C_MIN_MIN_MIN0018.jpg'

# Carregar a imagem
img = cv2.imread(img_path)

# Verificar se a imagem foi carregada corretamente
if img is None:
    print(f"Erro ao carregar a imagem. Verifique o caminho do arquivo: {img_path}")
else:
    print("Imagem carregada com sucesso!")

    # Pré-processamento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Suavizar a imagem
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Usar o detector de bordas Canny
    edges = cv2.Canny(blurred, threshold1=25, threshold2=50)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Preparar a figura para animação
    fig, ax = plt.subplots()

    # Criar uma imagem em branco para desenhar os contornos progressivamente
    img_canvas = np.copy(img)

    # Desativar a grade e os eixos
    ax.set_axis_off()

    # Função de atualização da animação
    def update(frame):
        # Copiar a imagem de fundo
        img_copy = np.copy(img_canvas)

        # Desenhar os contornos até o frame atual
        cv2.drawContours(img_copy, contours[:frame], -1, (0, 255, 0), 2)  # Desenhar em verde

        # Mostrar a imagem no gráfico (convertendo de BGR para RGB)
        ax.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))

        # Atualizar título com o número do frame
        ax.set_title(f'Frame {frame}')

    # Criar animação
    ani = animation.FuncAnimation(fig, update, frames=len(contours)+1, interval=100)

    # Exibir a animação
    plt.show()
