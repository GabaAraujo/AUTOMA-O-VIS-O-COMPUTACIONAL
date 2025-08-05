import cv2
import numpy as np
import torch
from ultralytics import YOLO
import math
import sys
import os

class LaneCarDistanceSystem:
    def __init__(self, use_ufld=True):
        # Inicializar modelo YOLO para carros
        self.yolo_model = YOLO('yolov8n.pt')  # Será baixado automaticamente
        
        # Configurar UFLD para lane detection
        self.use_ufld = use_ufld
        if use_ufld:
            try:
                # Tentar importar UFLD
                from model.model import parsingNet
                
                # Carregar modelo UFLD pré-treinado
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.ufld_model = parsingNet(
                    pretrained=False, 
                    backbone='18',
                    cls_dim=(100+1, 56, 4),
                    use_aux=False
                ).to(device)
                
                # Carregar pesos pré-treinados
                try:
                    state_dict = torch.load('tusimple_18.pth', map_location=device)
                    compatible_state_dict = {}
                    for k, v in state_dict.items():
                        if 'module.' in k:
                            compatible_state_dict[k[7:]] = v
                        else:
                            compatible_state_dict[k] = v
                    self.ufld_model.load_state_dict(compatible_state_dict, strict=False)
                    self.ufld_model.eval()
                    print("UFLD carregado com sucesso!")
                except FileNotFoundError:
                    print("Arquivo tusimple_18.pth não encontrado. Usando detecção simples.")
                    self.use_ufld = False
                
            except Exception as e:
                print(f"Erro ao carregar UFLD: {e}")
                print("Usando detecção simples de faixas com OpenCV")
                self.use_ufld = False
        
        # Parâmetros da câmera (precisam ser calibrados para sua câmera específica)
        self.camera_height = 1.2  # Altura da câmera em metros
        self.camera_pitch = 10    # Ângulo da câmera em graus
        self.focal_length = 800   # Distância focal em pixels (aproximado)
        
        # Parâmetros da imagem
        self.image_height = 480
        self.image_width = 640
        
        # Altura média de um carro (metros)
        self.average_car_height = 1.5
        
        # Zona de segurança (metros)
        self.safe_distance = 20.0
        
    def detect_cars(self, frame):
        """Detecta carros usando YOLO"""
        results = self.yolo_model(frame, classes=[2])  # Classe 2 = car no COCO dataset
        
        cars = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                if confidence > 0.5:  # Filtrar detecções com baixa confiança
                    cars.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                        'bottom_y': int(y2)  # Importante para cálculo de distância
                    })
        
        return cars
    
    def detect_lanes_ufld(self, frame):
        """Detecta faixas usando Ultra-Fast-Lane-Detection"""
        # Preprocessar imagem para UFLD
        img = cv2.resize(frame, (800, 288))
        img = img.astype(np.float32) / 255.0
        
        # Normalização ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Converter para tensor PyTorch
        img = img.transpose(2, 0, 1)  # HWC para CHW
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # Garantir tipo float32
        
        device = next(self.ufld_model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        # Inferência
        with torch.no_grad():
            output = self.ufld_model(img_tensor)
        
        # Mover output para CPU e converter para numpy
        if isinstance(output, tuple):
            output = output[0]  # Pegar apenas o primeiro elemento se for tupla
        output = output.cpu().numpy()
        
        # Processar output para obter coordenadas das faixas
        col_sample = np.linspace(0, 800 - 1, 100)
        
        lanes = []
        # Ajustar baseado na estrutura real do output
        if len(output.shape) >= 3:
            num_lanes = min(4, output.shape[1])  # Máximo 4 faixas
            for i in range(num_lanes):
                lane_points = []
                for j in range(min(100, output.shape[2])):
                    # Verificar se o ponto é válido
                    prob = output[0, i, j] if len(output.shape) > 2 else output[0, j]
                    if prob > 0.5:  # Threshold para considerar o ponto válido
                        x = int(col_sample[j] if j < len(col_sample) else col_sample[-1])
                        y = int(prob * 288) if prob <= 1.0 else int(prob)
                        
                        # Converter para coordenadas da imagem original
                        x_orig = int(x * frame.shape[1] / 800)
                        y_orig = int(y * frame.shape[0] / 288)
                        
                        # Validar coordenadas
                        if 0 <= x_orig < frame.shape[1] and 0 <= y_orig < frame.shape[0]:
                            lane_points.append((x_orig, y_orig))
                
                if len(lane_points) > 5:  # Filtrar faixas com poucos pontos
                    lanes.append(lane_points)
        
        return lanes
    
    def detect_lanes(self, frame):
        """Detecta faixas - usa UFLD se disponível, senão OpenCV"""
        if self.use_ufld:
            try:
                return self.detect_lanes_ufld(frame)
            except Exception as e:
                print(f"Erro no UFLD, mudando para detecção simples: {e}")
                self.use_ufld = False
                return self.detect_lanes_simple(frame)
        else:
            return self.detect_lanes_simple(frame)
    
    def detect_lanes_simple(self, frame):
        """Detecção simples de faixas usando Canny + Hough Transform (fallback)"""
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur gaussiano
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detecção de bordas
        edges = cv2.Canny(blur, 50, 150)
        
        # Região de interesse (parte inferior da imagem)
        height, width = edges.shape
        roi = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (0, int(height * 0.6)),
            (width, int(height * 0.6)),
            (width, height)
        ]], np.int32)
        cv2.fillPoly(roi, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, roi)
        
        # Detecção de linhas usando Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=100,
            maxLineGap=50
        )
        
        lane_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Filtrar linhas muito horizontais
                if abs(y2 - y1) > 10:
                    lane_lines.append([(x1, y1), (x2, y2)])
        
        return lane_lines
    
    def estimate_distance_simple(self, car):
        """Estima distância usando a posição Y do carro na imagem"""
        bottom_y = car['bottom_y']
        
        # Método 1: Baseado na altura do carro na imagem
        x1, y1, x2, y2 = car['bbox']
        car_height_pixels = y2 - y1
        
        # Fórmula aproximada: distância = (altura_real * focal_length) / altura_pixels
        if car_height_pixels > 0:
            distance_method1 = (self.average_car_height * self.focal_length) / car_height_pixels
        else:
            distance_method1 = float('inf')
        
        # Método 2: Baseado na posição Y (perspectiva)
        # Quanto mais embaixo na imagem, mais perto está
        y_normalized = (self.image_height - bottom_y) / self.image_height
        distance_method2 = 5 + (y_normalized * 50)  # Entre 5 e 55 metros aproximadamente
        
        # Usar média dos dois métodos para maior robustez
        if distance_method1 < 100:  # Filtrar valores muito altos
            distance = (distance_method1 + distance_method2) / 2
        else:
            distance = distance_method2
            
        return max(1.0, min(distance, 100.0))  # Limitar entre 1 e 100 metros
    
    def get_car_lane_position(self, car, lanes):
        """Determina em qual lado da faixa o carro está"""
        car_center_x = car['center'][0]
        image_center = self.image_width / 2
        
        if car_center_x < image_center - 50:
            return "Faixa Esquerda"
        elif car_center_x > image_center + 50:
            return "Faixa Direita" 
        else:
            return "Faixa Central"
    
    def analyze_safety(self, cars):
        """Analisa riscos de segurança baseado na distância"""
        alerts = []
        
        for i, car in enumerate(cars):
            distance = car.get('distance', float('inf'))
            lane_position = car.get('lane_position', 'Desconhecida')
            
            # Verificar se está muito próximo
            if distance < self.safe_distance:
                risk_level = "ALTO" if distance < 10 else "MÉDIO"
                alerts.append({
                    'car_id': i,
                    'distance': distance,
                    'lane': lane_position,
                    'risk': risk_level,
                    'message': f"Carro na {lane_position} a {distance:.1f}m - Risco {risk_level}"
                })
        
        return alerts
    
    def draw_results(self, frame, cars, lanes, alerts):
        """Desenha os resultados na imagem"""
        result_frame = frame.copy()
        
        # Desenhar faixas
        if lanes:
            for lane in lanes:
                if isinstance(lane[0], tuple):  # Se são pontos individuais (UFLD)
                    for i in range(len(lane) - 1):
                        cv2.line(result_frame, lane[i], lane[i+1], (0, 255, 0), 2)
                else:  # Se são linhas (detecção simples)
                    pt1, pt2 = lane
                    cv2.line(result_frame, pt1, pt2, (0, 255, 0), 3)
        
        # Desenhar carros e informações
        for i, car in enumerate(cars):
            x1, y1, x2, y2 = car['bbox']
            distance = car.get('distance', 0)
            lane_position = car.get('lane_position', 'N/A')
            
            # Cor baseada na distância
            if distance < 10:
                color = (0, 0, 255)  # Vermelho - muito próximo
            elif distance < 20:
                color = (0, 165, 255)  # Laranja - próximo
            else:
                color = (0, 255, 0)  # Verde - seguro
            
            # Desenhar bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Informações do carro
            info_text = f"Dist: {distance:.1f}m"
            cv2.putText(result_frame, info_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Posição da faixa
            cv2.putText(result_frame, lane_position, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Desenhar alertas
        y_offset = 30
        for alert in alerts:
            cv2.putText(result_frame, alert['message'], (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
        
        # Adicionar informações sobre o método de detecção usado
        method_text = "UFLD" if self.use_ufld else "OpenCV"
        cv2.putText(result_frame, f"Lane Detection: {method_text}", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def process_frame(self, frame):
        """Processa um frame completo"""
        # Redimensionar frame se necessário
        frame = cv2.resize(frame, (self.image_width, self.image_height))
        
        # 1. Detectar carros
        cars = self.detect_cars(frame)
        
        # 2. Detectar faixas
        lanes = self.detect_lanes(frame)
        
        # 3. Calcular distâncias e posições
        for car in cars:
            car['distance'] = self.estimate_distance_simple(car)
            car['lane_position'] = self.get_car_lane_position(car, lanes)
        
        # 4. Analisar segurança
        alerts = self.analyze_safety(cars)
        
        # 5. Desenhar resultados
        result_frame = self.draw_results(frame, cars, lanes, alerts)
        
        return result_frame, cars, lanes, alerts

# Exemplo de uso
def main():
    # Inicializar sistema
    # Definir use_ufld=False se não tiver UFLD configurado ou estiver com problemas
    system = LaneCarDistanceSystem(use_ufld=True)
    
    # Para vídeo de arquivo
    video_path = 'video/car.mp4'  # Substitua pelo seu vídeo
    
    if not os.path.exists(video_path):
        print(f"Arquivo de vídeo não encontrado: {video_path}")
        print("Tentando usar câmera padrão...")
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o vídeo ou câmera")
        return
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fim do vídeo ou erro na captura")
                break
            
            frame_count += 1
            
            # Processar frame
            try:
                result_frame, cars, lanes, alerts = system.process_frame(frame)
                
                # Mostrar resultado
                cv2.imshow('Lane Detection + Car Detection + Distance', result_frame)
                
                # Imprimir alertas no terminal (apenas a cada 30 frames para reduzir spam)
                if frame_count % 30 == 0:
                    print(f"\n--- Frame {frame_count} ---")
                    print(f"Carros detectados: {len(cars)}")
                    print(f"Faixas detectadas: {len(lanes)}")
                    for alert in alerts:
                        print(alert['message'])
                
            except Exception as e:
                print(f"Erro ao processar frame {frame_count}: {e}")
                continue
            
            # Sair com 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()