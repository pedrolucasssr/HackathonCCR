import numpy as np
import cv2
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from calibration import load_calibration
from copy import copy


class Lane():
    def __init__(self):
        # Detecta se linha foi detectada no ultimo frame ou nao
        self.detected = False
        # Valores em x para linha detectada em pixels
        self.cur_fitx = None
        # Valores em y para linha detectada em pixels
        self.cur_fity = None
        # valores x para ultimas N ajustes de linha
        self.prev_fitx = []
        # Coeficientes polinomiais para ajuste mais recente
        self.current_poly = [np.array([False])]
        # Melhor coeficiente polinomial para ultima iteracao
        self.prev_poly = [np.array([False])]

    def average_pre_lanes(self):
        tmp = copy(self.prev_fitx)
        tmp.append(self.cur_fitx)
        self.mean_fitx = np.mean(tmp, axis=0)

    def append_fitx(self):
        if len(self.prev_fitx) == N:
            self.prev_fitx.pop(0)
        self.prev_fitx.append(self.mean_fitx)

    def process(self, ploty):
        self.cur_fity = ploty
        self.average_pre_lanes()
        self.append_fitx()
        self.prev_poly = self.current_poly


left_lane = Lane()
right_lane = Lane()
frame_width = 1280
frame_height = 720

LANEWIDTH = 3.6  # largura de faixa no Brasil: 3.6 meters
input_scale = 1
output_frame_scale = 1
N = 4 # carregar N linhas previas

# tamanho de imagem:1280x720
x = [194, 1117, 705, 575]
y = [719, 719, 461, 461]
X = [290, 990, 990, 290]
Y = [719, 719, 0, 0]

src = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]]) / input_scale)
dst = np.floor(np.float32([[X[0], Y[0]], [X[1], Y[1]],[X[2], Y[2]], [X[3], Y[3]]]) / input_scale)

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

# Para criar visualizacao final do video
X_b = [574, 706, 706, 574]
Y_b = [719, 719, 0, 0]
src_ = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]]) / (input_scale*2))
dst_ = np.floor(np.float32([[X_b[0], Y_b[0]], [X_b[1], Y_b[1]],[X_b[2], Y_b[2]], [X_b[3], Y_b[3]]]) / (input_scale*2))
M_b = cv2.getPerspectiveTransform(src_, dst_)

# Limite para cor e gradiente
s_thresh, sx_thresh, dir_thresh, m_thresh, r_thresh = (120, 255), (20, 100), (0.7, 1.3), (30, 100), (200, 255)

# carrega calibracao
calib_file = 'calibration_pickle.p'
mtx, dist = load_calibration(calib_file)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Os seguintes processos sao realizados a img
    # 1- converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2- Puxa a derivativa em x ou y dada a orientacao
    # 3- Toma valor absoluto da derivada
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 4- Escala para 8-bit (0-255) e converte o tipo de imagem para np.uint8
    scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))

    # 5 Cria a mascara de escala do gradiente
    # thresh_min < VALOR < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):

    # Os seguintes processos sao realizados a img
    # 1- converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2- Puxa o gradiente de x e y separadamente
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3- Calcula a magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # 4- Escala para 8-bit (0-255) e converte o tipo de imagem para np.uint8
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # 5- Cria a mascara binaria onde os limites mag sao encontrados
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Limites de acordo com a direcao do gradiente
    :param img:
    :param sobel_kernel:
    :param thresh:
    :return:
    """

    # Os seguintes processos sao realizados a img
    # 1- converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2- Puxa o gradiente de x e y separadamente
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3- Pega os valores absolutos gradiente de de x e y
    # 4- usa np.arctan2(abs_sobely, abs_sobelx) para calcular a direcao do gradiente
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # 5- Cria a mascara binaria onde encontra os limites de direcao
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def gradient_pipeline(image, ksize = 3, sx_thresh=(20, 100), sy_thresh=(20, 100), m_thresh=(30, 100), dir_thresh=(0.7, 1.3)):

    # Aplica as funcoes de limite
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=sx_thresh)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=sy_thresh)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=m_thresh)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=dir_thresh)
    combined = np.zeros_like(mag_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # combined[(gradx == 1)  | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def threshold_col_channel(channel, thresh):

    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

    return binary



def find_edges(img, s_thresh=s_thresh, sx_thresh=sx_thresh, dir_thresh=dir_thresh):

    img = np.copy(img)
    # Converte para HSV color space e coloca como limite o canal s
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    s_binary = threshold_col_channel(s_channel, thresh=s_thresh)

    # Sobel em x
    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)
    # mag_binary = mag_thresh(img, sobel_kernel=3, thresh=m_thresh)
    # direcao do gradiente
    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=dir_thresh)
    # mascara de saida
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(( (sxbinary == 1) & (dir_binary==1) ) | ( (s_binary == 1) & (dir_binary==1) ))] = 1

    # adiciona mais pesos para o canal s
    c_bi = np.zeros_like(s_channel)
    c_bi[( (sxbinary == 1) & (s_binary==1) )] = 2

    ave_binary = (combined_binary + c_bi)

    return ave_binary


def warper(img, M):

    # Computa e aplica a transformada de perspectiva
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # mantem mesmo tamanho de imagem de entrada

    return warped


# Encaixa na linha de pista
def full_search(binary_warped, visualization=False):

    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Cria imagem de saida para desenhar e visualizar o resultado
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = out_img.astype('uint8')

    # Encontra o pico das metades da esquerda e direita do histograma
    # Essas linhas serão o ponto de partida para as linhas da esquerda e da direita
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Escolhe o número de janelas deslizantes
    nwindows = 9
    # Define altura das janelas
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identifica posicoes x e y de todos os pixels nao zero
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # posicoes atuais a serem atualizadas em cada janela
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Define largura de janelas e margem
    margin = np.floor(100/input_scale)
    # Define numero minimo de pixels encontrados para recentralizar janela
    minpix = np.floor(50/input_scale)
    # Cria listas vazias para receber indices de pixels de faixa esquerdos e direitos
    left_lane_inds = []
    right_lane_inds = []

    # Passa por cada janela
    for window in range(nwindows):
        # Identifica fronteiras em x e y (esquerda e direita)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Desenha na janela de visualizacao de imagem
        if visualization:
            cv2.rectangle(out_img,(int(win_xleft_low),int(win_y_low)),(int(win_xleft_high),int(win_y_high)),(0,255,0), 2)
            cv2.rectangle(out_img,(int(win_xright_low),int(win_y_low)),(int(win_xright_high),int(win_y_high)),(0,255,0), 2)

        # Identifica os pixels nao zero em x e y na janela
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Acrescentaos indices as listas
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # Se localizados mais pixels minpix, recentralizar proxima janela na posicao principal
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatena indices de matrizes
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extrai posicoes dos pixels de linha direitos e esquerdos
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Ajusta polinomio de segunda ordem a cada um
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Visualizacao

    # Gera valores x e y para plotagem
    if visualization:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.subplot(1,2,1)
        plt.imshow(out_img)
        # plt.imshow(binary_warped)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim((0, frame_width / input_scale))
        plt.ylim((frame_height / input_scale, 0))
        plt.show()

    return left_fit, right_fit



def window_search(left_fit, right_fit, binary_warped, margin=100, visualization=False):
    # Encontra linhas de pixel com a busca por janela
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Novamente extrai posicoes dos pixels de linha direitos e esquerdos
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Ajusta polinomio de segunda ordem a cada um
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if visualization:
        # Gera valores x e y para plotagem
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Aqui termina o tratamento da imagem. Os proximos passos servem para visualizar os resultados
        # Cria uma imagem para desenhar e mostrar na janela de selecao
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img = out_img.astype('uint8')
        window_img = np.zeros_like(out_img)
        # Cor nos pixels das linhas esquerda e direita
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Gera um poligono para ilustrar a area de busca da janela
        # Tambem reformular pontos x e y para formatos usaveis para cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Desenha pista na imagem deformada
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim((0, frame_width / input_scale))
        plt.ylim((frame_height / input_scale, 0))

        plt.show()

    return left_fit, right_fit


def measure_lane_curvature(ploty, leftx, rightx, visualization=False):

    # Reversao para encaixar topo a base em x e y
    leftx = leftx[::-1]
    rightx = rightx[::-1]

    # Escolhe valor maximo de y, correspondente a base da imagem
    y_eval = np.max(ploty)

    # Define conversao de espaco em x e y de pixel para metro
    ym_per_pix = 30/(frame_height/input_scale)
    xm_per_pix = LANEWIDTH/(700/input_scale)

    # Ajusta novos polinomios para x e y no espaco
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calcula novo raio de curvatura
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Agora o raio de curvatura esta em metros
    # print(left_curverad, 'm', right_curverad, 'm')

    if leftx[0] - leftx[-1] > 50/input_scale:
        curve_direction = 'Left curve'
    elif leftx[-1] - leftx[0] > 50/input_scale:
        curve_direction = 'Right curve'
    else:
        curve_direction = 'Straight'

    return (left_curverad+right_curverad)/2.0, curve_direction


def off_center(left, mid, right):
    """
    Define se esta fora do eixo de acordo com posicao do eixo central do veiculo
    em relação a posicao da faixa da direita e esquerda
    :param left: posicao da faixa da esquerda 
    :param mid:  posicao do veiculo
    :param right: posicao da faixa da direita
    :return: True or False, indicador de off-center
    """
    a = mid - left
    b = right - mid
    width = right - left

    if a >= b:  # Perdendo o eixo para direita
        offset = a / width * LANEWIDTH - LANEWIDTH /2.0
    else:       # Perdendo o eixo para esquerda
        offset = LANEWIDTH /2.0 - b / width * LANEWIDTH

    return offset


def compute_car_offcenter(ploty, left_fitx, right_fitx, undist):

    # Cria imagem para desenho das linhas
    height = undist.shape[0]
    width = undist.shape[1]

    # Reformula pontos x e y para formatos usaveis para cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    bottom_l = left_fitx[height-1]
    bottom_r = right_fitx[0]

    offcenter = off_center(bottom_l, width/2.0, bottom_r)

    return offcenter, pts


def create_output_frame(offcenter, pts, undist_ori, fps, curvature, curve_direction, binary_sub, threshold=0.6):
    """
    :param offcenter:
    :param pts:
    :param undist_ori:
    :param fps:
    :param threshold:
    :return:
    """

    undist_ori = cv2.resize(undist_ori, (0,0), fx=1/output_frame_scale, fy=1/output_frame_scale)
    w = undist_ori.shape[1]
    h = undist_ori.shape[0]

    undist_birdview = warper(cv2.resize(undist_ori, (0,0), fx=1/2, fy=1/2), M_b)

    color_warp = np.zeros_like(undist_ori).astype(np.uint8)

    # Cria frame para cada imagem
    whole_frame = np.zeros((int(h*2.5), int(w*2.34), 3), dtype=np.uint8)


    if abs(offcenter) > threshold:  # Se o veiculo esta fora do centro mais de 0.6m
        # Faixa vermelha
        cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0)) # red
    else: # Faixa verde
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))  # green

    newwarp = cv2.warpPerspective(color_warp, M_inv, (int(frame_width/input_scale), int(frame_height/input_scale)))

    # Combina resultado com imagem original  
    # result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    newwarp_ = cv2.resize(newwarp,None, fx=input_scale/output_frame_scale, fy=input_scale/output_frame_scale, interpolation = cv2.INTER_LINEAR)

    output = cv2.addWeighted(undist_ori, 1, newwarp_, 0.3, 0)

    ############## Gera saida combinada para visualizacao apenas ################
    # whole_frame[40:40+h, 20:20+w, :] = undist_ori
    # whole_frame[40:40+h, 60+w:60+2*w, :] = output
    # whole_frame[220+h/2:220+2*h/2, 20:20+w/2, :] = undist_birdview
    # whole_frame[220+h/2:220+2*h/2, 40+w/2:40+w, 0] = cv2.resize((binary_sub*255).astype(np.uint8), (0,0), fx=1/2, fy=1/2)
    # whole_frame[220+h/2:220+2*h/2, 40+w/2:40+w, 1] = cv2.resize((binary_sub*255).astype(np.uint8), (0,0), fx=1/2, fy=1/2)
    # whole_frame[220+h/2:220+2*h/2, 40+w/2:40+w, 2] = cv2.resize((binary_sub*255).astype(np.uint8), (0,0), fx=1/2, fy=1/2)
    #
    # font = cv2.FONT_HERSHEY_SIMPLEX
    if offcenter >= 0:
        offset = offcenter
        direction = 'Right'
    elif offcenter < 0:
        offset = -offcenter
        direction = 'Left'

    lane_info = {'curvature': curvature, 'curve_direction': curve_direction, 'dev_dir': direction, 'offset': offset}

    return whole_frame, output, lane_info


def tracker(binary_sub, ploty, visualization=False):
    # 

    left_fit, right_fit = window_search(left_lane.prev_poly, right_lane.prev_poly, binary_sub, margin=100/input_scale, visualization=visualization)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    std_value = np.std(right_fitx - left_fitx)
    if std_value < (85 /input_scale):
        left_lane.detected = True
        right_lane.detected = True
        left_lane.current_poly = left_fit
        right_lane.current_poly = right_fit
        left_lane.cur_fitx = left_fitx
        right_lane.cur_fitx = right_fitx
    else:
        left_lane.detected = False
        right_lane.detected = False
        left_lane.current_poly = left_lane.prev_poly
        right_lane.current_poly = right_lane.prev_poly
        left_lane.cur_fitx = left_lane.prev_fitx[-1]
        right_lane.cur_fitx = right_lane.prev_fitx[-1]


def detector(binary_sub, ploty, visualization=False):

    left_fit, right_fit = full_search(binary_sub, visualization=visualization)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    std_value = np.std(right_fitx - left_fitx)
    if std_value < (85 /input_scale):
        left_lane.current_poly = left_fit
        right_lane.current_poly = right_fit
        left_lane.cur_fitx = left_fitx
        right_lane.cur_fitx = right_fitx
        left_lane.detected = True
        right_lane.detected = True
    else:
        left_lane.current_poly = left_lane.prev_poly
        right_lane.current_poly = right_lane.prev_poly
        if len(left_lane.prev_fitx) > 0:
            left_lane.cur_fitx = left_lane.prev_fitx[-1]
            right_lane.cur_fitx = right_lane.prev_fitx[-1]
        else:
            left_lane.cur_fitx = left_fitx
            right_lane.cur_fitx = right_fitx
        left_lane.detected = False
        right_lane.detected = False



def lane_process(img, visualization=False):

    start = timer()
    # Redimensiona imagem de entrada de acordo com escala
    img_undist_ = cv2.undistort(img, mtx, dist, None, mtx)
    img_undist = cv2.resize(img_undist_, (0,0), fx=1/input_scale, fy=1/input_scale)

    # Encontra imagem binaria de faixa/borda
    img_binary = find_edges(img_undist)

    # Deforma imagem para birdview
    binary_warped = warper(img_binary, M)  # get binary image contains edges

    # Corta imagem binaria
    binary_sub = np.zeros_like(binary_warped)
    binary_sub[:, int(150/input_scale):int(-80/input_scale)]  = binary_warped[:, int(150/input_scale):int(-80/input_scale)]

    # Inicia detector/tracker para encontrar faixa
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    if left_lane.detected:  # tracker
        tracker(binary_sub, ploty, visualization)
    else:  # detector
        detector(binary_sub, ploty, visualization)

    # Media entre os N frames anteriores para gerar a faixa media
    left_lane.process(ploty)
    right_lane.process(ploty)

    # Mede curvatura da faixa
    curvature, curve_direction = measure_lane_curvature(ploty, left_lane.mean_fitx, right_lane.mean_fitx)

    # Computa decentralizacao do carro em metros
    offcenter, pts = compute_car_offcenter(ploty, left_lane.mean_fitx, right_lane.mean_fitx, img_undist)

    # Computa taxa de processamento de frames
    end = timer()
    fps = 1.0 / (end - start)

    # Combina todas as imagens em um video de saida final para visualização
    _, single_view, lane_info = create_output_frame(offcenter, pts, img_undist_, fps, curvature, curve_direction, binary_sub)
    return img_undist_, single_view, lane_info
