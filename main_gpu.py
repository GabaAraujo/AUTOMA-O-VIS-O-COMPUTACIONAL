import cv2
import numpy as np
import torch
from ultralytics import YOLO
import math
import sys
import os
import scipy.special
import time

class LaneCarDistanceSystem:
    def __init__(self, use_ufld=True, use_gpu=True):
        print("🚀 Inicializando sistema com GPU e UFLD...")
        
        # Configuração de GPU/CUDA
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        if self.use_gpu:
            print(f"🎮 Usando GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            # Otimizações CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            print(f"💻 Usando CPU")
        
        # Parâmetros para processamento UFLD
        self.ufld_input_size = (800, 288)
        self.num_lanes = 4
        self.num_points = 100
        self.griding_num = 100
        
        # Inicializar modelo YOLO para carros
        print("📦 Carregando modelo YOLO...")
        self.yolo_model = YOLO('yolov8n.pt')
        if self.use_gpu:
            self.yolo_model.to('cuda')
        
        # Configurar UFLD para lane detection
        self.use_ufld = use_ufld
        self.ufld_model = None
        
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
        
        # Cache para otimização UFLD
        self.col_sample = np.linspace(0, 800 - 1, self.griding_num)
        self.col_sample_w = self.col_sample[1] - self.col_sample[0]
        self.idx = np.arange(self.griding_num) + 1
        self.idx = self.idx.reshape(-1, 1, 1)
        
        # Inicializar UFLD se solicitado
        if use_ufld:
            self._init_ufld()
        
    def _init_ufld(self):
        """Inicializa o modelo UFLD"""
        try:
            # Verificar se o diretório do UFLD existe
            if not os.path.exists('model'):
                raise ImportError("Diretório 'model' não encontrado. Certifique-se de ter clonado o repositório UFLD.")
            
            # Importar modelo UFLD
            from model.model import parsingNet
            
            # Criar modelo com configuração correta para TuSimple
            self.ufld_model = parsingNet(
                pretrained=False, 
                backbone='18',
                cls_dim=(self.griding_num + 1, 56, self.num_lanes),  # 56 para TuSimple
                use_aux=False
            ).to(self.device)
            
            # Carregar pesos pré-treinados
            weight_file = 'tusimple_18.pth'
            if not os.path.exists(weight_file):
                raise FileNotFoundError(f"""
Arquivo {weight_file} não encontrado!

Para baixar os pesos:
1. Vá para: https://github.com/cfzd/Ultra-Fast-Lane-Detection
2. Baixe o arquivo tusimple_18.pth da seção de releases
3. Coloque o arquivo na pasta raiz do projeto

Ou use o comando:
wget https://github.com/cfzd/Ultra-Fast-Lane-Detection/releases/download/v1.0.0/tusimple_18.pth
                """)
            
            print(f"📥 Carregando pesos de {weight_file}...")
            state_dict = torch.load(weight_file, map_location=self.device)
            
            # Verificar se o state_dict tem a chave 'model'
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Limpar nomes das chaves se necessário
            compatible_state_dict = {}
            for k, v in state_dict.items():
                if 'module.' in k:
                    compatible_state_dict[k[7:]] = v
                else:
                    compatible_state_dict[k] = v
            
            self.ufld_model.load_state_dict(compatible_state_dict, strict=False)
            self.ufld_model.eval()
            
            print("✓ UFLD inicializado com sucesso!")
            
            # Mostrar informações de memória se usando GPU
            if self.use_gpu:
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                print(f"💾 Memória GPU alocada: {memory_allocated:.1f} MB")
                print(f"💾 Memória GPU reservada: {memory_reserved:.1f} MB")
            
        except ImportError as e:
            print(f"❌ Erro ao importar UFLD: {e}")
            print("Usando detecção simples de faixas com OpenCV")
            self.use_ufld = False
        
        except Exception as e:
            print(f"❌ Erro ao inicializar UFLD: {e}")
            print("Usando detecção simples de faixas com OpenCV")
            self.use_ufld = False
    
    def detect_cars(self, frame):
        """Detecta carros usando YOLO"""
        results = self.yolo_model(frame, classes=[2], verbose=False)  # Classe 2 = car
        
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
    
    def preprocess_for_ufld(self, frame):
        """Preprocessa imagem para UFLD"""
        # Redimensionar para tamanho esperado pelo UFLD
        img = cv2.resize(frame, self.ufld_input_size)
        
        # Converter BGR para RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalizar para [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Normalização ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Converter HWC para CHW
        img = img.transpose(2, 0, 1)
        
        # Converter para tensor PyTorch
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        
        return img_tensor.to(self.device)
    
    def postprocess_ufld_output(self, output, original_shape):
        """Processa output do UFLD para obter coordenadas das faixas - VERSÃO OTIMIZADA"""
        if self.ufld_model is None:
            return []
        
        # Importar constantes do TuSimple
        from data.constant import tusimple_row_anchor
        
        # Mover para CPU se necessário
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        elif isinstance(output, tuple):
            output = output[0].cpu().numpy()
        
        # Dimensões originais da imagem
        orig_h, orig_w = original_shape[:2]
        
        # Processar saída do modelo (baseado no demo.py) - VERSÃO OTIMIZADA
        out_j = output[0]  # Pegar primeiro batch
        out_j = out_j[:, ::-1, :]  # Inverter
        
        # Aplicar softmax
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        
        # Calcular localização usando cache
        loc = np.sum(prob * self.idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == self.griding_num] = 0
        out_j = loc
        
        lanes = []
        
        # Processar cada faixa
        for i in range(out_j.shape[1]):
            lane_points = []
            
            # Verificar se a faixa tem pontos suficientes
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        # Calcular coordenadas usando cache
                        x_coord = int(out_j[k, i] * self.col_sample_w * orig_w / 800) - 1
                        y_coord = int(orig_h * (tusimple_row_anchor[56-1-k]/288)) - 1
                        
                        # Validar coordenadas
                        if 0 <= x_coord < orig_w and 0 <= y_coord < orig_h:
                            lane_points.append((x_coord, y_coord))
                
                # Adicionar faixa se tiver pontos suficientes
                if len(lane_points) > 3:
                    # Ordenar pontos por Y (de cima para baixo)
                    lane_points.sort(key=lambda p: p[1])
                    lanes.append(lane_points)
        
        return lanes
    
    def detect_lanes_ufld(self, frame):
        """Detecta faixas usando Ultra-Fast-Lane-Detection"""
        if self.ufld_model is None:
            raise RuntimeError("Modelo UFLD não foi inicializado corretamente")
        
        # Preprocessar imagem
        input_tensor = self.preprocess_for_ufld(frame)
        
        # Inferência
        with torch.no_grad():
            output = self.ufld_model(input_tensor)
        
        # Pós-processar output
        lanes = self.postprocess_ufld_output(output, frame.shape)
        
        return lanes
    
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
    
    def detect_lanes(self, frame):
        """Detecta faixas - usa UFLD se disponível, senão OpenCV"""
        if self.use_ufld and self.ufld_model is not None:
            try:
                return self.detect_lanes_ufld(frame)
            except Exception as e:
                print(f"Erro no UFLD, mudando para detecção simples: {e}")
                return self.detect_lanes_simple(frame)
        else:
            return self.detect_lanes_simple(frame)
    
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
        car_bottom_y = car['bottom_y']
        
        if not lanes:
            # Fallback para posição simples se não há faixas detectadas
            image_center = self.image_width / 2
            if car_center_x < image_center - 50:
                return "Faixa Esquerda"
            elif car_center_x > image_center + 50:
                return "Faixa Direita"
            else:
                return "Faixa Central"
        
        # Encontrar faixas próximas ao carro
        lane_x_positions = []
        for lane in lanes:
            # Encontrar ponto da faixa mais próximo ao bottom do carro
            closest_point = None
            min_distance = float('inf')
            
            for point in lane:
                if isinstance(point, tuple):
                    x, y = point
                    distance = abs(y - car_bottom_y)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = x
            
            if closest_point is not None:
                lane_x_positions.append(closest_point)
        
        if lane_x_positions:
            # Calcular posição relativa às faixas
            avg_lane_x = sum(lane_x_positions) / len(lane_x_positions)
            if car_center_x < avg_lane_x - 30:
                return "Faixa Esquerda"
            elif car_center_x > avg_lane_x + 30:
                return "Faixa Direita"
            else:
                return "Faixa Central"
        else:
            # Fallback
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
        method_text = "UFLD" if self.use_ufld and self.ufld_model is not None else "OpenCV"
        device_text = "GPU" if self.use_gpu else "CPU"
        cv2.putText(result_frame, f"Lane Detection: {method_text} | Device: {device_text}", 
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
    print("=== Sistema de Detecção de Faixas e Carros (GPU + UFLD) ===")
    
    try:
        # Verificar GPU
        if torch.cuda.is_available():
            print(f"🎮 GPU detectada: {torch.cuda.get_device_name(0)}")
            print(f"💾 Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("💻 Usando CPU (GPU não disponível)")
        
        # Inicializar sistema
        # Definir use_ufld=True para usar UFLD, use_gpu=True para usar GPU
        system = LaneCarDistanceSystem(use_ufld=True, use_gpu=True)
        
        # Para vídeo de arquivo
        video_path = 'video/car2.mp4'  # Substitua pelo seu vídeo
        
        if not os.path.exists(video_path):
            print(f"⚠️ Arquivo de vídeo não encontrado: {video_path}")
            print("Tentando usar câmera padrão...")
            cap = cv2.VideoCapture(0)
        else:
            print(f"📹 Carregando vídeo: {video_path}")
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Erro: Não foi possível abrir o vídeo ou câmera")
            return
        
        # Obter informações do vídeo
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📊 FPS do vídeo: {fps_video:.1f}")
        print(f"📊 Total de frames: {total_frames}")
        
        print("🚀 Sistema iniciado! Pressione 'q' para sair, 'p' para pausar.")
        
        frame_count = 0
        total_cars = 0
        total_lanes = 0
        processing_times = []
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("📺 Fim do vídeo ou erro na captura")
                break
            
            frame_count += 1
            
            try:
                # Processar frame
                result_frame, cars, lanes, alerts = system.process_frame(frame)
                
                # Calcular tempo de processamento
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                current_fps = 1.0 / processing_time if processing_time > 0 else 0
                
                # Estatísticas
                total_cars += len(cars)
                total_lanes += len(lanes)
                
                # Adicionar informações de performance ao frame
                avg_fps = 1.0 / (sum(processing_times[-30:]) / len(processing_times[-30:])) if len(processing_times) > 0 else 0
                
                # Informações de performance
                perf_info = [
                    f"FPS: {current_fps:.1f} (Avg: {avg_fps:.1f})",
                    f"Frame: {frame_count}/{total_frames}",
                    f"Carros: {len(cars)} | Faixas: {len(lanes)}",
                    f"Tempo: {processing_time*1000:.1f}ms"
                ]
                
                # Desenhar informações de performance
                for i, info in enumerate(perf_info):
                    cv2.putText(result_frame, info, (10, 30 + i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Mostrar resultado
                cv2.imshow('Lane Detection + Car Detection + Distance (GPU)', result_frame)
                
                # Log periódico
                if frame_count % 60 == 0:  # A cada 60 frames
                    avg_cars = total_cars / frame_count
                    avg_lanes = total_lanes / frame_count
                    avg_processing_time = sum(processing_times[-60:]) / len(processing_times[-60:])
                    avg_fps_log = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                    
                    print(f"\n📊 Frame {frame_count}")
                    print(f"   FPS médio: {avg_fps_log:.1f}")
                    print(f"   Tempo médio: {avg_processing_time*1000:.1f}ms")
                    print(f"   Média carros/frame: {avg_cars:.1f}")
                    print(f"   Média faixas/frame: {avg_lanes:.1f}")
                    
                    if alerts:
                        print("   🚨 Alertas ativos:")
                        for alert in alerts:
                            print(f"     {alert['message']}")
                    
                    # Mostrar uso de memória GPU se disponível
                    if system.use_gpu:
                        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                        print(f"   💾 GPU Mem: {memory_allocated:.1f}MB / {memory_reserved:.1f}MB")
                
            except Exception as e:
                print(f"❌ Erro ao processar frame {frame_count}: {e}")
                continue
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):  # Pausar
                print("⏸️ Pausado. Pressione qualquer tecla para continuar...")
                cv2.waitKey(0)
            elif key == ord('s'):  # Salvar frame
                cv2.imwrite(f'frame_{frame_count}.jpg', result_frame)
                print(f"💾 Frame {frame_count} salvo!")
                
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        print("\n📋 Verifique:")
        print("1. UFLD está instalado corretamente")
        print("2. Arquivo tusimple_18.pth está presente")
        print("3. Dependências do PyTorch estão instaladas")
        
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        
        # Relatório final
        if frame_count > 0:
            print(f"\n📈 RELATÓRIO FINAL:")
            print(f"   Total de frames processados: {frame_count}")
            print(f"   FPS médio: {1.0 / (sum(processing_times) / len(processing_times)):.1f}")
            print(f"   Carros detectados: {total_cars}")
            print(f"   Faixas detectadas: {total_lanes}")
            print(f"   Dispositivo usado: {'GPU' if system.use_gpu else 'CPU'}")
            print(f"   Método de detecção: {'UFLD' if system.use_ufld and system.ufld_model else 'OpenCV'}")
        
        print("🏁 Sistema encerrado.")

if __name__ == "__main__":
    main()