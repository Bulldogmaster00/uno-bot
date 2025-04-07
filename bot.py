import cv2
import numpy as np
import pytesseract
from collections import defaultdict

class UNOCardDetector:
    def __init__(self):
        self.color_ranges = {
            'red': [
                ((0, 120, 70), (10, 255, 255)),
                ((170, 120, 70), (180, 255, 255))
            ],
            'blue': ((90, 120, 70), (140, 255, 255)),
            'green': ((36, 50, 50), (89, 255, 255)),
            'yellow': ((20, 100, 100), (35, 255, 255)),
            'black': ((0, 0, 0), (180, 50, 50))
        }
        self.special_cards = {
            'S': 'skip',
            'R': 'reverse',
            '+2': 'draw_two',
            'W': 'wild',
            '+4': 'wild_draw_four'
        }

    def detect_color(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_scores = defaultdict(int)

        for color, ranges in self.color_ranges.items():
            if color == 'red':
                mask = cv2.bitwise_or(
                    cv2.inRange(hsv, np.array(ranges[0][0]), np.array(ranges[0][1])),
                    cv2.inRange(hsv, np.array(ranges[1][0]), np.array(ranges[1][1]))
                )
            else:
                mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))

            color_scores[color] = cv2.countNonZero(mask)

        dominant_color = max(color_scores, key=color_scores.get)
        return dominant_color if color_scores[dominant_color] > 1000 else 'unknown'

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    def detect_card_type(self, img):
        processed = self.preprocess_image(img)
        config = r'--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789SRW+24'
        text = pytesseract.image_to_string(processed, config=config).strip()

        for symbol, name in self.special_cards.items():
            if symbol in text:
                return name

        clean_text = ''.join([c for c in text if c.isdigit()])
        if clean_text:
            number = int(clean_text)
            if 0 <= number <= 9:
                return number
        return None

    def detect_card(self, img):
        h, w = img.shape[:2]
        cropped = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]

        color = self.detect_color(cropped)
        value = self.detect_card_type(cropped)

        if color != 'black' and color != 'unknown':
            if isinstance(value, int):
                return {'type': 'number', 'color': color, 'value': value}
            elif value in ['skip', 'reverse', 'draw_two']:
                return {'type': 'action', 'color': color, 'action': value}
        elif color == 'black':
            if value in ['wild', 'wild_draw_four']:
                return {'type': 'wild', 'action': value}
            return {'type': 'wild', 'action': 'wild'}

        return None

class UNOGame:
    def __init__(self):
        self.detector = UNOCardDetector()
        self.player_hand = []
        self.top_card = None
        self.current_color = None

    def capture_card(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access camera")
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            cv2.putText(frame, "C - Capture | Q - Cancel | ESC - Menu", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('UNO Card Capture', frame)

            key = cv2.waitKey(1)
            if key == 27:
                print("\n⏪ Capture canceled. Returning to menu.")
                cv2.destroyAllWindows()
                return None
            elif key in [ord('c'), ord('C')]:
                card_img = frame[h//4:3*h//4, w//4:3*w//4]
                cv2.destroyAllWindows()
                break
            elif key in [ord('q'), ord('Q')]:
                cv2.destroyAllWindows()
                return None

        cap.release()
        return self.detector.detect_card(card_img)

    def melhores_jogadas(self):
        if not self.top_card:
            print("Nenhuma carta no topo da pilha.")
            return

        jogaveis = []
        for carta in self.player_hand:
            if carta['type'] == 'wild':
                jogaveis.append(carta)
            elif carta['color'] == self.top_card.get('color'):
                jogaveis.append(carta)
            elif carta['type'] == self.top_card['type']:
                if carta['type'] == 'number' and carta['value'] == self.top_card.get('value'):
                    jogaveis.append(carta)
                elif carta['type'] == 'action' and carta['action'] == self.top_card.get('action'):
                    jogaveis.append(carta)

        jogaveis.sort(key=lambda c: 0 if c['type'] == 'number' else (1 if c['type'] == 'action' else 2))

        print("\n💡 Sugestão de jogadas possíveis:")
        if jogaveis:
            for j in jogaveis:
                if j['type'] == 'number':
                    print(f"➡️ {j['color']} {j['value']}")
                elif j['type'] == 'action':
                    print(f"➡️ {j['color']} {j['action']}")
                elif j['type'] == 'wild':
                    print(f"➡️ {j['action']}")
        else:
            print("❌ Nenhuma jogada possível.")

    def remove_card_from_hand(self):
        if not self.player_hand:
            print("\n❌ A mão está vazia. Nenhuma carta para remover.")
            return

        print("\n🗑️  Mão atual:")
        for idx, c in enumerate(self.player_hand):
            if c['type'] == 'number':
                print(f"{idx+1}: {c['color']} {c['value']}")
            elif c['type'] == 'action':
                print(f"{idx+1}: {c['color']} {c['action']}")
            elif c['type'] == 'wild':
                print(f"{idx+1}: {c['action']}")

        try:
            choice = int(input("Digite o número da carta que deseja remover (0 para cancelar): "))
            if choice == 0:
                print("⏪ Remoção cancelada.")
                return
            if 1 <= choice <= len(self.player_hand):
                removed = self.player_hand.pop(choice - 1)
                print(f"✅ Carta removida: {removed}")
            else:
                print("❌ Número inválido.")
        except ValueError:
            print("❌ Entrada inválida. Por favor, insira um número.")

    def run(self):
        while True:
            print("\n🎮 UNO Game Menu:")
            print("1. 📷 Capturar carta da câmera")
            print("2. ⌨️  Inserir carta manualmente como carta do topo")
            print("3. 🖐️  Ver mão do jogador")
            print("4. ➕ Adicionar carta à mão")
            print("5. 🔄 Mudar carta do topo da pilha")
            print("6. 🗑️  Remover carta da mão")
            print("7. ❌ Sair")
            choice = input("Escolha uma opção: ")

            if choice == '1':
                card = self.capture_card()
                if card:
                    self.top_card = card
                    print("\n🃏 Carta detectada como carta do topo:", card)
                    self.melhores_jogadas()
            elif choice == '2':
                card_input = input("Digite a carta (ex: red 5, blue skip, wild_draw_four): ").strip()
                if card_input == '':
                    print("\n⏪ Entrada vazia. Voltando ao menu.")
                    continue
                card = self.parse_manual_input(card_input, adicionar_na_mao=False)
                if card:
                    self.top_card = card
                    print("\n🃏 Carta definida como carta do topo:", card)
                    self.melhores_jogadas()
                else:
                    print("\n❌ Entrada inválida. Tente novamente.")
            elif choice == '3':
                print("\n🖐️ Mão atual:")
                for idx, c in enumerate(self.player_hand):
                    print(f"{idx+1}: {c}")
                input("Pressione Enter para voltar ao menu.")
            elif choice == '4':
                card_input = input("Digite a carta para adicionar à mão: ").strip()
                if card_input == '':
                    print("\n⏪ Entrada vazia. Voltando ao menu.")
                    continue
                card = self.parse_manual_input(card_input, adicionar_na_mao=True)
                if card:
                    print("\n🃏 Carta adicionada à mão:", card)
                else:
                    print("\n❌ Entrada inválida. Tente novamente.")
            elif choice == '5':
                card_input = input("Digite a nova carta do topo (ex: red 5, blue skip, wild_draw_four): ").strip()
                if card_input == '':
                    print("\n⏪ Entrada vazia. Voltando ao menu.")
                    continue
                card = self.parse_manual_input(card_input, adicionar_na_mao=False)
                if card:
                    self.top_card = card
                    print("\n🃏 Carta do topo atualizada:", card)
                    self.melhores_jogadas()
                else:
                    print("\n❌ Entrada inválida. Tente novamente.")
            elif choice == '6':
                self.remove_card_from_hand()
            elif choice == '7':
                print("👋 Saindo do jogo. Até mais!")
                break
            else:
                print("Opção inválida. Tente novamente.")

    def parse_manual_input(self, text, adicionar_na_mao=True):
        parts = text.lower().split()
        if len(parts) == 2:
            color, value = parts
            if color in ['red', 'blue', 'green', 'yellow']:
                if value.isdigit() and 0 <= int(value) <= 9:
                    card = {'type': 'number', 'color': color, 'value': int(value)}
                elif value in ['skip', 'reverse', 'draw_two']:
                    card = {'type': 'action', 'color': color, 'action': value}
                else:
                    return None
                if adicionar_na_mao:
                    self.player_hand.append(card)
                return card
        elif len(parts) == 1:
            if parts[0] in ['wild', 'wild_draw_four']:
                card = {'type': 'wild', 'action': parts[0]}
                if adicionar_na_mao:
                    self.player_hand.append(card)
                return card
        return None

if __name__ == "__main__":
    try:
        pytesseract.get_tesseract_version()
    except:
        print("ERROR: Tesseract OCR not properly installed")
        print("Install from https://github.com/tesseract-ocr/tesseract")
        exit()

    game = UNOGame()
    game.run()
