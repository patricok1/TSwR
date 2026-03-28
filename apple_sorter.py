import cv2
import numpy as np
import math
import os
import snap7
from snap7.util import *
import sys 
import json

# --- ZMIENNE GLOBALNE DLA KONFIGURACJI ---
KONFIGURACJA = {}

# --- USTAWIENIA ANALIZY ---
CZAS_OCZEKIWANIA_MS = 100 
WSPOLCZYNNIK_SKALI = 0.4 
WSPOLCZYNNIK_SKALI_MONTAZU = 0.2 
MIN_POWIERZCHNIA_USZKODZENIA_PROG = 25 
WSPOLCZYNNIK_KALIBRACJI_CM_NA_PIKSEL = {'W': 0.0, 'H': 0.0}

# --- KONSTANTY KATEGORYZACJI ---
PROG_KOLORU_PROCENT_DOMINACJI = 50.0 

# --- KONFIGURACJA PLC ---
IP_PLC = '192.168.0.4' 
RACK_PLC = 0
SLOT_PLC = 1
DB_KOLOR = 1 
DB_OBJETOSC = 2 
DB_USZKODZENIA = 3 
DB_NUMER_ELEMENTU = 4 
KOD_KOLORU_BEZ_AKCJI = 5 
DB_RED_PCT = 5      # Nowy rejestr dla średniej ważonej czerwieni [%]
DB_YELLOW_PCT = 6   # Nowy rejestr dla średniej ważonej żółci [%]

# --- FUNKCJA: ŁADOWANIE KONFIGURACJI Z JSON ---

def wczytaj_konfiguracje(plik_konfiguracyjny):
    """Wczytuje progi kolorów i inne parametry z pliku JSON z pełnej ścieżki."""
    global KONFIGURACJA
    
    # 1. Definicja twardo zakodowanych wartości (jako fallback)
    KONFIGURACJA_FALLBACK = {
        "color_thresholds": { 
            "LAB": {
                # ZMODYFIKOWANO DLA CIEMNIEJSZEJ CZERWIENI
                "RED": {"lower": [10, 135, 130], "upper": [255, 255, 255]},
                "YELLOW": {"lower": [100, 100, 120], "upper": [255, 165, 255]}
            },
            "HSV": {
                "RED_1": {"lower": [0, 40, 40], "upper": [10, 255, 255]},
                "RED_2": {"lower": [160, 40, 40], "upper": [180, 255, 255]},
                "YELLOW": {"lower": [15, 40, 60], "upper": [45, 255, 255]}
            },
            "BGR": {
                "RED": {"lower": [0, 0, 80], "upper": [140, 140, 255]},
                "YELLOW": {"lower": [0, 100, 100], "upper": [120, 255, 255]}  
            },

            "SEGMENTACJA": {
                "LAB_L_MIN": 70,
                "HSV_S_MIN": 50,
                "ROZMIAR_KERNELA": 5
            }
        }
    }

    try:
        with open(plik_konfiguracyjny, 'r') as f:
            dane_wczytane = json.load(f)
            
            if "color_thresholds" in dane_wczytane:
                KONFIGURACJA["progi_kolorow"] = dane_wczytane["color_thresholds"]
            else:
                raise KeyError("Brak klucza 'color_thresholds' w pliku konfiguracyjnym.")

        print(f"Konfiguracja wczytana pomyślnie z {plik_konfiguracyjny}.")
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku konfiguracyjnego: {plik_konfiguracyjny}. Używam twardo zakodowanych wartości.")
        KONFIGURACJA["progi_kolorow"] = KONFIGURACJA_FALLBACK["color_thresholds"]
    except KeyError as e:
        print(f"BŁĄD STRUKTURY: {e}. Używam twardo zakodowanych wartości rezerwowych.")
        KONFIGURACJA["progi_kolorow"] = KONFIGURACJA_FALLBACK["color_thresholds"]

# --- FUNKCJE POMOCNICZE ---

def wyczysc_terminal():
    """Wyczyszczenie terminala przed rozpoczeciem analizy."""
    os.system('cls' if os.name == 'nt' else 'clear')

def rysuj_tekst_z_tlem(obraz, tekst, x, y):
    """Funkcja pomocnicza do umieszczania tekstu z tlem."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255) 
    bg_color = (0, 0, 0) 
    (text_width, text_height), baseline = cv2.getTextSize(tekst, font, 0.7, 2)
    cv2.rectangle(obraz, (x, y - text_height - baseline), (x + text_width, y), bg_color, cv2.FILLED)
    cv2.putText(obraz, tekst, (x, y - baseline), font, 0.7, color, 2, cv2.LINE_AA)

def pobierz_maske_koloru_lab(obraz_lab, nazwa_koloru):
    dane_koloru = KONFIGURACJA['progi_kolorow']['LAB'].get(nazwa_koloru)
    if dane_koloru:
        lower = np.array(dane_koloru['lower'])
        upper = np.array(dane_koloru['upper'])
        return cv2.inRange(obraz_lab, lower, upper)
    else:
        return np.zeros(obraz_lab.shape[:2], dtype=np.uint8)


def pobierz_maske_koloru_hsv_bgr(obraz, nazwa_koloru, przestrzen_koloru):
    
    if przestrzen_koloru == 'HSV':
        
        if nazwa_koloru == 'RED':
            red1_data = KONFIGURACJA['progi_kolorow']['HSV'].get('RED_1')
            red2_data = KONFIGURACJA['progi_kolorow']['HSV'].get('RED_2')
            
            if red1_data and red2_data:
                lower_red1 = np.array(red1_data['lower'])
                upper_red1 = np.array(red1_data['upper'])
                lower_red2 = np.array(red2_data['lower'])
                upper_red2 = np.array(red2_data['upper'])
                mask1 = cv2.inRange(obraz, lower_red1, upper_red1)
                mask2 = cv2.inRange(obraz, lower_red2, upper_red2)
                return cv2.bitwise_or(mask1, mask2)
                
        else:
            dane_koloru = KONFIGURACJA['progi_kolorow']['HSV'].get(nazwa_koloru)
            if dane_koloru:
                lower = np.array(dane_koloru['lower'])
                upper = np.array(dane_koloru['upper'])
                return cv2.inRange(obraz, lower, upper)
            
    elif przestrzen_koloru == 'BGR':
        dane_koloru = KONFIGURACJA['progi_kolorow']['BGR'].get(nazwa_koloru)
        if dane_koloru:
            lower = np.array(dane_koloru['lower'])
            upper = np.array(dane_koloru['upper'])
            return cv2.inRange(obraz, lower, upper)
            
   


def wyswietl_montaz_masek(grupy_masek, procenty, nazwa_wideo, ramka_oryginalna):
    """Tworzy i wyswietla okno z maskami kolorów i uszkodzeń. Zmieniono z 4x3 na 3x3."""
    
    H, W = ramka_oryginalna.shape[:2]
    # Montaz 3x3: RED, YELLOW, OTHER
    montaz = np.zeros((H * 3, W * 3, 3), dtype=np.uint8)
    
    nazwy_kolorow_wiersze = ['RED', 'YELLOW', 'OTHER'] 
    nazwy_przestrzeni = ['LAB', 'HSV', 'BGR']
    
    for row_idx, nazwa_koloru in enumerate(nazwy_kolorow_wiersze):
        for col_idx, nazwa_przestrzeni in enumerate(nazwy_przestrzeni):
            
            widok_maskowany = np.zeros_like(ramka_oryginalna)
            
            if nazwa_koloru == 'OTHER':
                maska = grupy_masek[nazwa_przestrzeni]['OTHER']
                widok_maskowany[maska > 0] = [0, 0, 255] if nazwa_przestrzeni != 'LAB' else [128, 128, 128]
            else:
                # RED lub YELLOW
                maska = grupy_masek[nazwa_przestrzeni][nazwa_koloru]
                widok_maskowany = cv2.bitwise_and(ramka_oryginalna, ramka_oryginalna, mask=maska)
            
            y1, x1 = row_idx * H, col_idx * W
            y2, x2 = y1 + H, x1 + W
            
            # Wklejamy do montażu 3x3
            if y2 <= montaz.shape[0]:
                montaz[y1:y2, x1:x2] = widok_maskowany
            
            # Tekst
            if nazwa_koloru == 'OTHER':
                procent = procenty['LAB_JEDNOZNACZNE']['other_pct']
                tekst_procentowy = f"LAB INNE/USZK: {procent:.1f}%" if nazwa_przestrzeni == 'LAB' else f"{nazwa_przestrzeni} INNE/USZK."
            else:
                if nazwa_przestrzeni == 'LAB':
                     procent = procenty['LAB_JEDNOZNACZNE'][f"{nazwa_koloru.lower()}_pct"]
                     tekst_procentowy = f"LAB {nazwa_koloru}: {procent:.1f}%" 
                else:
                     # Pokazujemy procent koloru dla danej przestrzeni
                     avg_pct_key = f"{nazwa_koloru.lower()}_pct"
                     avg_pct_value = procenty[f'{nazwa_przestrzeni}_JEDNOZNACZNE'][avg_pct_key]
                     tekst_procentowy = f"{nazwa_przestrzeni} {nazwa_koloru}: {avg_pct_value:.1f}%"
                
            rysuj_tekst_z_tlem(montaz, tekst_procentowy, x1 + 10, y1 + 30)

    skalowany_montaz = cv2.resize(montaz, None, fx=WSPOLCZYNNIK_SKALI_MONTAZU, fy=WSPOLCZYNNIK_SKALI_MONTAZU, 
                                 interpolation=cv2.INTER_LINEAR)
                                 
    cv2.imshow('Montaz Masek 3x3 (LAB | HSV | BGR)', skalowany_montaz)


def pobierz_wspolczynnik_kalibracji(zrodlo_kalibracji, znana_szerokosc_cm, znana_wysokosc_cm):

    
    seg_config = KONFIGURACJA['progi_kolorow']['SEGMENTACJA']
    
    # PARAMETRY SEGMANTACJI DLA WYKRYCIA KONTURU (SKORYGOWANE)
    L_min_kontur = 60  # Dolny próg jasności LAB L (odcina cienie)
    L_max_kontur = 140 # Górny próg jasności LAB L (odcina jasne, szare tło)
    rozmiar_kernela = seg_config['ROZMIAR_KERNELA']
    min_pole_jablka = 100 # Minimalna powierzchnia konturu
    
    # --- ZMIANA: Wczytanie POJEDYNCZEGO ZDJĘCIA ---
    klatka = cv2.imread(zrodlo_kalibracji)
    
    if klatka is None:
        print(f"BŁĄD: Nie można wczytać pliku kalibracyjnego: {zrodlo_kalibracji}. Używam domyślnej wartości.")
        return {'W': 0.0125, 'H': 0.0125}
    
    # Przetwarzanie jednej klatki (zdjęcia)
    ramka_przeskalowana = cv2.resize(klatka, None, fx=WSPOLCZYNNIK_SKALI, fy=WSPOLCZYNNIK_SKALI, interpolation=cv2.INTER_LINEAR)
    lab_przeskalowane = cv2.cvtColor(ramka_przeskalowana, cv2.COLOR_BGR2LAB) 
    
    # --- SEGMENTACJA KONTURU (TYLKO NA PODSTAWIE JASNOŚCI L) ---
    L_kanal = lab_przeskalowane[:,:,0] 
    
    _, maska_jablka_L_min = cv2.threshold(L_kanal, L_min_kontur, 255, cv2.THRESH_BINARY)
    _, maska_jablka_L_max = cv2.threshold(L_kanal, L_max_kontur, 255, cv2.THRESH_BINARY_INV)
    maska_jablka = cv2.bitwise_and(maska_jablka_L_min, maska_jablka_L_max)
    
    # Operacje morfologiczne (oczyszczanie maski)
    kernel = np.ones((rozmiar_kernela, rozmiar_kernela), np.uint8)
    maska_jablka = cv2.morphologyEx(maska_jablka, cv2.MORPH_OPEN, kernel, iterations=3)
    maska_jablka = cv2.morphologyEx(maska_jablka, cv2.MORPH_CLOSE, kernel, iterations=5) 
    
    kontury, _ = cv2.findContours(maska_jablka, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    przefiltrowane_kontury = [c for c in kontury if cv2.contourArea(c) > min_pole_jablka]
    
    if przefiltrowane_kontury:
        najwiekszy_kontur = max(przefiltrowane_kontury, key=cv2.contourArea)
        prostokat = cv2.minAreaRect(najwiekszy_kontur)
        
        szerokosc_skala = max(prostokat[1]); wysokosc_skala = min(prostokat[1])
        
        szerokosc_oryg = szerokosc_skala / WSPOLCZYNNIK_SKALI
        wysokosc_oryg = wysokosc_skala / WSPOLCZYNNIK_SKALI
        
        # --- KOREKTA DLA WYSOKICH/WĄSKICH OBIEKTÓW (jak karton) ---
        # Zakładamy, że W/H w cm podane w konfiguracji są poprawne.
        # Wymieniamy wartości pikselowe, aby pasowały do logicznej interpretacji.
        if szerokosc_oryg > wysokosc_oryg and znana_szerokosc_cm < znana_wysokosc_cm:
            # Wymiana, jeśli obraz sugeruje szeroki, a pomiary wąski obiekt
            szerokosc_final = wysokosc_oryg
            wysokosc_final = szerokosc_oryg
        elif szerokosc_oryg < wysokosc_oryg and znana_szerokosc_cm > znana_wysokosc_cm:
            # Wymiana, jeśli obraz sugeruje wąski, a pomiary szeroki obiekt
            szerokosc_final = wysokosc_oryg
            wysokosc_final = szerokosc_oryg
        else:
            szerokosc_final = szerokosc_oryg
            wysokosc_final = wysokosc_oryg
            
        # Obliczenie współczynników z jednej klatki
        wspolczynnik_w = znana_szerokosc_cm / szerokosc_final
        wspolczynnik_h = znana_wysokosc_cm / wysokosc_final
       
# --- NOWA SEKCJA: WIZUALIZACJA KALIBRACJI ---
        skrzynka = np.intp(cv2.boxPoints(prostokat))
        obraz_podgladu = ramka_przeskalowana.copy()
        
        # Rysowanie ramki i konturu
        cv2.drawContours(obraz_podgladu, [skrzynka], 0, (0, 255, 0), 2)
        cv2.drawContours(obraz_podgladu, [najwiekszy_kontur], -1, (255, 0, 0), 1)
        
        # Dodanie tekstu z wymiarami w pikselach (na obrazie przeskalowanym)
        tekst_w = f"W: {szerokosc_oryg:.0f}px ({znana_szerokosc_cm}cm)"
        tekst_h = f"H: {wysokosc_oryg:.0f}px ({znana_wysokosc_cm}cm)"
        
        # Wyświetlenie informacji na obrazie
        rysuj_tekst_z_tlem(obraz_podgladu, "KALIBRACJA", 10, 30)
        rysuj_tekst_z_tlem(obraz_podgladu, tekst_w, 10, 60)
        rysuj_tekst_z_tlem(obraz_podgladu, tekst_h, 10, 90)
        
        cv2.imshow('Podglad Kalibracji - Nacisnij dowolny klawisz', obraz_podgladu)
        cv2.waitKey(0) # Czeka na reakcję użytkownika
        cv2.destroyWindow('Podglad Kalibracji - Nacisnij dowolny klawisz')

        print(f" Kalibracja: Wykryto W: {szerokosc_final:.0f} px, H: {wysokosc_final:.0f} px")
        
        return {'W': wspolczynnik_w, 'H': wspolczynnik_h}
    else:
        print("Brak wykrytego konturu. Uzywam domyslnej wartosci.")
        return {'W': 0.0125, 'H': 0.0125}


def klasyfikuj_jablko(procenty_koloru, min_wymiar_cm):
    """
    Klasyfikacja (bez ufnosci) na podstawie średniej WAZONEJ koloru, uszkodzeń i rozmiaru.
    """
    
    # Wagi: LAB (7), HSV (2), BGR (1). Suma = 10
    WAGI = {'LAB': 7.0, 'HSV': 2.0, 'BGR': 1.0}
    SUMA_WAG = 10.0
    
    # 1. Obliczanie średnich WAŻONYCH wartości procentowych
    
    srednia_red = ((procenty_koloru['LAB_JEDNOZNACZNE']['red_pct'] * WAGI['LAB']) + 
                    (procenty_koloru['HSV_JEDNOZNACZNE']['red_pct'] * WAGI['HSV']) + 
                    (procenty_koloru['BGR_JEDNOZNACZNE']['red_pct'] * WAGI['BGR'])) / SUMA_WAG
    
    srednia_yellow = ((procenty_koloru['LAB_JEDNOZNACZNE']['yellow_pct'] * WAGI['LAB']) + 
                      (procenty_koloru['HSV_JEDNOZNACZNE']['yellow_pct'] * WAGI['HSV']) + 
                      (procenty_koloru['BGR_JEDNOZNACZNE']['yellow_pct'] * WAGI['BGR'])) / SUMA_WAG
    
    srednia_innych = ((procenty_koloru['LAB_JEDNOZNACZNE']['other_pct'] * WAGI['LAB']) + 
                      (procenty_koloru['HSV_JEDNOZNACZNE']['other_pct'] * WAGI['HSV']) + 
                      (procenty_koloru['BGR_JEDNOZNACZNE']['other_pct'] * WAGI['BGR'])) / SUMA_WAG
    
    klasa_kategoria = "ODRZUT"
    
    # STAŁE DLA KLASYFIKACJI (w cm)
    MIN_WYMIAR_KLASA1 = 7.0 
    MIN_WYMIAR_KLASA2 = 6.0 
    MIN_WYMIAR_KLASA3 = 5.0 
    
    PROCENT_KOLORU_KLASA1 = 70.0
    PROCENT_KOLORU_KLASA2 = 60.0
    PROCENT_KOLORU_KLASA3 = 50.0 
    
    MAX_USZKODZEN_KLASA1 = 1.0 
    MAX_USZKODZEN_KLASA2 = 5.0 
    MAX_USZKODZEN_KLASA3 = 10.0 
    
    
    # 2. PRÓBA KLASYFIKACJI KLASY 1
    if min_wymiar_cm >= MIN_WYMIAR_KLASA1 and srednia_innych <= MAX_USZKODZEN_KLASA1:
        if srednia_red >= PROCENT_KOLORU_KLASA1:
            klasa_kategoria = "RED_KLASA1"
        elif srednia_yellow >= PROCENT_KOLORU_KLASA1:
            klasa_kategoria = "YELLOW_KLASA1"

    # 3. PRÓBA KLASYFIKACJI KLASY 2 (Jeśli nie KLASA 1)
    if klasa_kategoria == "ODRZUT" and \
       min_wymiar_cm >= MIN_WYMIAR_KLASA2 and \
       srednia_innych <= MAX_USZKODZEN_KLASA2:
        
        if srednia_red >= PROCENT_KOLORU_KLASA2:
            klasa_kategoria = "RED_KLASA2"
        elif srednia_yellow >= PROCENT_KOLORU_KLASA2:
            klasa_kategoria = "YELLOW_KLASA2"

    # 4. PRÓBA KLASYFIKACJI KLASY 3 (Jeśli nie KLASA 1 ani KLASA 2)
    if klasa_kategoria == "ODRZUT" and \
       min_wymiar_cm >= MIN_WYMIAR_KLASA3 and \
       srednia_innych <= MAX_USZKODZEN_KLASA3:
        
        if srednia_red >= PROCENT_KOLORU_KLASA3:
            klasa_kategoria = "RED_KLASA3"
        elif srednia_yellow >= PROCENT_KOLORU_KLASA3:
            klasa_kategoria = "YELLOW_KLASA3"
            
    # 5. OSTATNIE KATEGORIE
    if klasa_kategoria == "ODRZUT":
        # USZKODZONE - jeśli przekroczy próg dla K3
        if srednia_innych > MAX_USZKODZEN_KLASA3:
             return "USZKODZONE", 0
        
        # ODRZUT - jeśli nie spełnia warunków dla żadnej klasy i nie jest USZKODZONE
        return "ODRZUT", 0
        
    
    # Ufność jest zawsze 0
    return klasa_kategoria, 0


# --- FUNKCJE KOMUNIKACJI PLC (Snap7) ---

def pobierz_kod_koloru_plc(nazwa_kategorii):
    """Mapuje kategorię jabłka na kod PLC."""
    if nazwa_kategorii.startswith('RED_KLASA1'):
        return 1
    elif nazwa_kategorii.startswith('YELLOW_KLASA1'):
        return 3
    elif nazwa_kategorii.startswith('RED_KLASA2'):
        return 11
    elif nazwa_kategorii.startswith('YELLOW_KLASA2'):
        return 13
    elif nazwa_kategorii.startswith('RED_KLASA3'):
        return 21
    elif nazwa_kategorii.startswith('YELLOW_KLASA3'):
        return 23
    elif nazwa_kategorii == 'USZKODZONE':
        return 90
    else:
        return 4 

def resetuj_dane_plc_i_lampki(klient_plc):
    """ZERUJE WSZYSTKIE GŁÓWNE REJESTRY PLC."""
    try:
        klient_plc.db_write(DB_KOLOR, 0, bytearray([0, 0])) 
        
        bufor_zero_float = bytearray(4)
        set_real(bufor_zero_float, 0, 0.0) 
        klient_plc.db_write(DB_OBJETOSC, 0, bufor_zero_float)
        
        klient_plc.db_write(DB_USZKODZENIA, 0, bufor_zero_float)
        
        klient_plc.db_write(DB_NUMER_ELEMENTU, 0, bytearray([0, 0])) 
        
        return "ZEROWANIE KOMPLETNE (Wszystkie DB = 0)"
    except Exception as e:
        return f"BŁĄD ZEROWANIA DANYCH PLC: {e}"

def wyslij_sygnal_bez_akcji(klient_plc):
    """Wysyła kod 5 (IDLE/Nieaktywne) i zeruje DB2/DB3."""
    try:
        bufor_kolor_bez_akcji = bytearray(2)
        set_int(bufor_kolor_bez_akcji, 0, KOD_KOLORU_BEZ_AKCJI)
        klient_plc.db_write(DB_KOLOR, 0, bufor_kolor_bez_akcji)
        
        bufor_zero_float = bytearray(4)
        set_real(bufor_zero_float, 0, 0.0) 
        klient_plc.db_write(DB_OBJETOSC, 0, bufor_zero_float)
        klient_plc.db_write(DB_USZKODZENIA, 0, bufor_zero_float)
        
        return "SYGNAŁ IDLE (5) WYSŁANY"
    except Exception as e:
        return f"BŁĄD WYSYŁANIA IDLE: {e}"


def wyslij_dane_jablka_do_plc(klient_plc, kategoria, objetosci_cm3, procent_innych, procent_czerwony, procent_zolty, numer_elementu):
    kod_koloru = pobierz_kod_koloru_plc(kategoria)

    bufor_objetosci = bytearray(4)
    set_real(bufor_objetosci, 0, objetosci_cm3)
    
    bufor_uszkodzen = bytearray(4) 
    set_real(bufor_uszkodzen, 0, procent_innych) 
    
    bufor_koloru = bytearray(2)
    set_int(bufor_koloru, 0, kod_koloru)
    
    bufor_numeru = bytearray(2)
    set_int(bufor_numeru, 0, numer_elementu)

    bufor_czerwony = bytearray(4)
    set_real(bufor_czerwony, 0, procent_czerwony)
    
    bufor_zolty = bytearray(4)
    set_real(bufor_zolty, 0, procent_zolty)
    
    try:
        klient_plc.db_write(DB_KOLOR, 0, bufor_koloru)
        klient_plc.db_write(DB_OBJETOSC, 0, bufor_objetosci)
        klient_plc.db_write(DB_USZKODZENIA, 0, bufor_uszkodzen) 
        klient_plc.db_write(DB_NUMER_ELEMENTU, 0, bufor_numeru)
        klient_plc.db_write(DB_RED_PCT, 0, bufor_czerwony)    # NOWE
        klient_plc.db_write(DB_YELLOW_PCT, 0, bufor_zolty) 
            
        return f"Wysłano UŚREDNIONE: {kategoria} ({kod_koloru}), V: {objetosci_cm3:.1f},R: {procent_czerwony:.1f}%, Y: {procent_zolty:.1f}%, U: {procent_innych:.1f}%, Nr: {numer_elementu}"

    except Exception as e:
        return f"BŁĄD ZAPISU PLC: {e}"
        
def wyslij_odroczone_dane(klient_plc, wynik_odroczony):
    """Wysyła uśrednione dane z poprzedniej analizy (wynik N-1) do PLC."""
    if not wynik_odroczony:
        return "Brak opóźnionych danych do wysłania."
        
    kategoria = wynik_odroczony['kategoryzacja']['kategoria']
    objetosci_cm3 = wynik_odroczony['ksztalt']['volume_cm_cubed']
    procent_innych = wynik_odroczony['jakosc']['avg_other_pct']
    procent_red = wynik_odroczony['jakosc']['avg_red_pct']     # NOWE
    procent_yellow = wynik_odroczony['jakosc']['avg_yellow_pct'] # NOWE
    numer_elementu = wynik_odroczony['numer_elementu'] 
    
    if klient_plc and klient_plc.get_connected():
        status_wysylki = wyslij_dane_jablka_do_plc(klient_plc, kategoria, objetosci_cm3, procent_innych, procent_red, procent_yellow, numer_elementu)
        return f"OPÓŹNIONA WYSYŁKA (Jabłko {numer_elementu}): {status_wysylki}"
    else:
        return "Brak aktywnego połączenia PLC dla opóźnionej wysyłki."


# --- GLOWNA FUNKCJA ANALIZUJĄCA JEDNO WIDEO ---

def analizuj_pojedyncze_wideo(sciezka_wideo, wspolczynniki_skali, klient_plc=None, wynik_odroczony=None, indeks_elementu=0):
    
    cap = cv2.VideoCapture(sciezka_wideo)
    if not cap.isOpened():
        print(f"Blad: Nie mozna wczytac pliku wideo: {sciezka_wideo}")
        return {"file": sciezka_wideo, "status": "Analiza nie powiodla sie: Blad wczytywania pliku."}

    wszystkie_procenty = {
        'red_lab': [], 'yellow_lab': [], 'other_lab': [],
        'red_hsv': [], 'yellow_hsv': [],
        'red_bgr': [], 'yellow_bgr': []
    }
    
    wszystkie_metryki_ksztaltu = {'ratio': [], 'szerokosc': [], 'wysokosc': [], 'objetosc': [], 'powierzchnia': [], 'min_wymiar_cm': []}
    
    seg_config = KONFIGURACJA['progi_kolorow']['SEGMENTACJA']
    L_min = seg_config['LAB_L_MIN']
    S_min = seg_config['HSV_S_MIN']
    rozmiar_kernela = seg_config['ROZMIAR_KERNELA']
    
    omin_klatek = 1
    licznik_klatek = 0
    nazwa_wideo = os.path.basename(sciezka_wideo)
    aktualny_numer_elementu = indeks_elementu + 1 

    status_odroczonej_wysylki = "Brak danych N-1"
    if klient_plc and klient_plc.get_connected():
        if wynik_odroczony:
            status_odroczonej_wysylki = wyslij_odroczone_dane(klient_plc, wynik_odroczony)
        else:
            status_odroczonej_wysylki = wyslij_sygnal_bez_akcji(klient_plc)
        
    print(f"\n---> Analiza wideo: {aktualny_numer_elementu} ({nazwa_wideo}) <---")

    while cap.isOpened():
        ret, ramka = cap.read()
        if not ret: break

        licznik_klatek += 1
        if licznik_klatek % omin_klatek != 0: continue
            
        ramka_przeskalowana = cv2.resize(ramka, None, fx=WSPOLCZYNNIK_SKALI, fy=WSPOLCZYNNIK_SKALI, interpolation=cv2.INTER_LINEAR)
        hsv_przeskalowane = cv2.cvtColor(ramka_przeskalowana, cv2.COLOR_BGR2HSV)
        lab_przeskalowane = cv2.cvtColor(ramka_przeskalowana, cv2.COLOR_BGR2LAB) 
        
        # --- 1. SEGMENTACJA JABLKA ---
        L_kanal = lab_przeskalowane[:,:,0] 
        _, maska_jablka_L = cv2.threshold(L_kanal, L_min, 255, cv2.THRESH_BINARY)
        
        S_kanal = hsv_przeskalowane[:,:,1]
        _, maska_nasycenia = cv2.threshold(S_kanal, S_min, 255, cv2.THRESH_BINARY)
        maska_jablka = cv2.bitwise_and(maska_jablka_L, maska_nasycenia)
        
        kernel = np.ones((rozmiar_kernela, rozmiar_kernela), np.uint8)
        maska_jablka = cv2.morphologyEx(maska_jablka, cv2.MORPH_OPEN, kernel, iterations=1)
        maska_jablka = cv2.morphologyEx(maska_jablka, cv2.MORPH_CLOSE, kernel, iterations=3) 
        
        kontury, _ = cv2.findContours(maska_jablka, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_pole_jablka = 500
        przefiltrowane_kontury = [c for c in kontury if cv2.contourArea(c) > min_pole_jablka]
        
        if not przefiltrowane_kontury:
            cv2.imshow('Analiza Ksztaltu Jablka', ramka_przeskalowana)
            if cv2.waitKey(CZAS_OCZEKIWANIA_MS) & 0xFF == ord('q'): break
            continue
            
        najwiekszy_kontur = max(przefiltrowane_kontury, key=cv2.contourArea)
        maska_jablka_pojedyncza = np.zeros(maska_jablka.shape, dtype=np.uint8)
        cv2.drawContours(maska_jablka_pojedyncza, [najwiekszy_kontur], -1, 255, cv2.FILLED)
        suma_pikseli_jablka = cv2.countNonZero(maska_jablka_pojedyncza)
        
        # --- 2. ANALIZA KSZTAŁTU I OBJĘTOŚCI ---
        prostokat = cv2.minAreaRect(najwiekszy_kontur)
        skrzynka = np.intp(cv2.boxPoints(prostokat))
        
        # Zgodnie z heurystyką, bierzemy min/max jako W i H (dla jabłka/spłaszczenia)
        szerokosc_skala = max(prostokat[1]); wysokosc_skala = min(prostokat[1])
        pole = cv2.contourArea(najwiekszy_kontur); obwod = cv2.arcLength(najwiekszy_kontur, True)
        stosunek_hw = wysokosc_skala / szerokosc_skala if szerokosc_skala > 0 else 0
        wspolczynnik_okraglosci = (4 * np.pi * pole) / (obwod * obwod) if obwod > 0 else 0
        
        wszystkie_metryki_ksztaltu['ratio'].append(stosunek_hw)
        szerokosc_oryg = szerokosc_skala / WSPOLCZYNNIK_SKALI
        wysokosc_oryg = wysokosc_skala / WSPOLCZYNNIK_SKALI
        wszystkie_metryki_ksztaltu['szerokosc'].append(szerokosc_oryg); wszystkie_metryki_ksztaltu['wysokosc'].append(wysokosc_oryg) 

        a_px = szerokosc_oryg / 2.0
        b_px = wysokosc_oryg / 2.0
        
        objetosc_px = (4.0 / 3.0) * math.pi * (a_px**2) * b_px
        wszystkie_metryki_ksztaltu['objetosc'].append(objetosc_px)

        if a_px > b_px and a_px > 0:
            e = math.sqrt(1.0 - (b_px**2) / (a_px**2))
            if e > 0:
                 powierzchnia_px = 2 * math.pi * (a_px**2) + math.pi * (b_px**2 / e) * math.log((1.0 + e) / (1.0 - e))
            else: 
                 powierzchnia_px = 4 * math.pi * (a_px**2)
        else:
             powierzchnia_px = 4 * math.pi * (a_px**2) 
        
        wszystkie_metryki_ksztaltu['powierzchnia'].append(powierzchnia_px)
            
        # --- 3. ANALIZA KOLORÓW I ELIMINACJA NAKŁADANIA SIĘ MASEK ---
        
        nazwy_kolorow = ['RED', 'YELLOW', 'GREEN'] 
        grupy_masek = {'LAB': {}, 'HSV': {}, 'BGR': {}}
        procenty_koloru = {}
        
        maski_surowe = {'LAB': {}, 'HSV': {}, 'BGR': {}}
        
        for nazwa in nazwy_kolorow:
            if nazwa != 'GREEN':
                maski_surowe['LAB'][nazwa] = cv2.bitwise_and(pobierz_maske_koloru_lab(lab_przeskalowane, nazwa), maska_jablka_pojedyncza)
                maski_surowe['HSV'][nazwa] = cv2.bitwise_and(pobierz_maske_koloru_hsv_bgr(hsv_przeskalowane, nazwa, 'HSV'), maska_jablka_pojedyncza)
                maski_surowe['BGR'][nazwa] = cv2.bitwise_and(pobierz_maske_koloru_hsv_bgr(ramka_przeskalowana, nazwa, 'BGR'), maska_jablka_pojedyncza)
            else:
                empty_mask = np.zeros(maska_jablka_pojedyncza.shape, dtype=np.uint8)
                maski_surowe['LAB'][nazwa] = empty_mask
                maski_surowe['HSV'][nazwa] = empty_mask
                maski_surowe['BGR'][nazwa] = empty_mask


        finalne_liczniki_pikseli = {'LAB': {}, 'HSV': {}, 'BGR': {}}
        
        for cs in ['LAB', 'HSV', 'BGR']:
            maska_red_final = maski_surowe[cs]['RED'].copy()
            maska_yellow_final = cv2.subtract(maski_surowe[cs]['YELLOW'], maska_red_final)
            
            # Suma wykrytych kolorów
            maska_unia_kolorow = cv2.bitwise_or(maska_red_final, maska_yellow_final)
            
            # OTHER to wszystko, co jest jabłkiem, ale nie jest czerwone ani żółte
            maska_other_final = cv2.subtract(maska_jablka_pojedyncza, maska_unia_kolorow)
            
            grupy_masek[cs]['RED'] = maska_red_final
            grupy_masek[cs]['YELLOW'] = maska_yellow_final
            grupy_masek[cs]['OTHER'] = maska_other_final # Przypisanie maski dopełnienia
            
            finalne_liczniki_pikseli[cs]['RED'] = cv2.countNonZero(maska_red_final)
            finalne_liczniki_pikseli[cs]['YELLOW'] = cv2.countNonZero(maska_yellow_final)
            finalne_liczniki_pikseli[cs]['OTHER'] = cv2.countNonZero(maska_other_final)

        # 3.3 OBLICZENIE PROCENTÓW KOLORÓW
        
        procent_red_lab = (finalne_liczniki_pikseli['LAB']['RED'] / suma_pikseli_jablka) * 100
        procent_yellow_lab = (finalne_liczniki_pikseli['LAB']['YELLOW'] / suma_pikseli_jablka) * 100
        procent_green_lab = 0.0 
        
        suma_pikseli_lab = finalne_liczniki_pikseli['LAB']['RED'] + finalne_liczniki_pikseli['LAB']['YELLOW']
        piksele_inne = suma_pikseli_jablka - suma_pikseli_lab
        procent_innych = (piksele_inne / suma_pikseli_jablka) * 100
        
        wszystkie_procenty['red_lab'].append(procent_red_lab)
        wszystkie_procenty['yellow_lab'].append(procent_yellow_lab)
        wszystkie_procenty['other_lab'].append(procent_innych)
        
        procent_red_hsv = (finalne_liczniki_pikseli['HSV']['RED'] / suma_pikseli_jablka) * 100
        procent_yellow_hsv = (finalne_liczniki_pikseli['HSV']['YELLOW'] / suma_pikseli_jablka) * 100
        procent_green_hsv = 0.0 
        wszystkie_procenty['red_hsv'].append(procent_red_hsv)
        wszystkie_procenty['yellow_hsv'].append(procent_yellow_hsv)
        
        procent_red_bgr = (finalne_liczniki_pikseli['BGR']['RED'] / suma_pikseli_jablka) * 100
        procent_yellow_bgr = (finalne_liczniki_pikseli['BGR']['YELLOW'] / suma_pikseli_jablka) * 100
        procent_green_bgr = 0.0 
        wszystkie_procenty['red_bgr'].append(procent_red_bgr)
        wszystkie_procenty['yellow_bgr'].append(procent_yellow_bgr)
        
        procent_innych_hsv = (finalne_liczniki_pikseli['HSV']['OTHER'] / suma_pikseli_jablka) * 100
        procent_innych_bgr = (finalne_liczniki_pikseli['BGR']['OTHER'] / suma_pikseli_jablka) * 100

        aktualne_procenty_kolorow = {
            "LAB_JEDNOZNACZNE": {"red_pct": procent_red_lab, "yellow_pct": procent_yellow_lab, "other_pct": procent_innych},
            "HSV_JEDNOZNACZNE": {"red_pct": procent_red_hsv, "yellow_pct": procent_yellow_hsv, "other_pct": procent_innych_hsv},
            "BGR_JEDNOZNACZNE": {"red_pct": procent_red_bgr, "yellow_pct": procent_yellow_bgr, "other_pct": procent_innych_bgr},
        }


        # --- 4. OBLICZENIA I KATEGORYZACJA BIEŻĄCEJ RAMKI ---
        skala_w = wspolczynniki_skali['W']
        skala_h = wspolczynniki_skali['H']
        aktualna_szerokosc_cm = szerokosc_oryg * skala_w
        aktualna_wysokosc_cm = wysokosc_oryg * skala_h
        
        aktualna_objetosc_cm3 = objetosc_px * skala_w * skala_w * skala_h 
        aktualna_powierzchnia_cm2 = powierzchnia_px * skala_w * skala_h 
        
        min_wymiar_cm = min(aktualna_szerokosc_cm, aktualna_wysokosc_cm)
        # Dodajemy min wymiar do listy metryk
        wszystkie_metryki_ksztaltu['min_wymiar_cm'].append(min_wymiar_cm)
        
        aktualne_procenty_kolorow = {
            "LAB_JEDNOZNACZNE": {"red_pct": procent_red_lab, "green_pct": procent_green_lab, "yellow_pct": procent_yellow_lab, "other_pct": procent_innych},
            "HSV_JEDNOZNACZNE": {"red_pct": procent_red_hsv, "green_pct": procent_green_hsv, "yellow_pct": procent_yellow_hsv, "other_pct": 100.0 - (procent_red_hsv + procent_yellow_hsv)},
            "BGR_JEDNOZNACZNE": {"red_pct": procent_red_bgr, "green_pct": procent_green_hsv, "yellow_pct": procent_yellow_bgr, "other_pct": 100.0 - (procent_red_bgr + procent_yellow_bgr)},
        }
        
        # Używamy tylko uśrednionego min. wymiaru do klasyfikacji na końcu.
        aktualna_kategoria, _ = klasyfikuj_jablko(aktualne_procenty_kolorow, min_wymiar_cm) 
        
        if klient_plc and klient_plc.get_connected():
            status_plc = "Polaczono"
        else:
            status_plc = "Brak polaczenia"
        
        # --- 6. WIZUALIZACJA ---
        
        cv2.drawContours(ramka_przeskalowana, [skrzynka], 0, (255, 0, 0), 2)
        cv2.drawContours(ramka_przeskalowana, [najwiekszy_kontur], -1, (0, 255, 255), 1)

        y_offset = 20
        
        rysuj_tekst_z_tlem(ramka_przeskalowana, f"ID: {nazwa_wideo}", 10, y_offset)
        y_offset += 20

        rysuj_tekst_z_tlem(ramka_przeskalowana, f"W: {szerokosc_oryg:.0f} px | H: {wysokosc_oryg:.0f} px", 10, y_offset)
        y_offset += 20
        rysuj_tekst_z_tlem(ramka_przeskalowana, f"W: {aktualna_szerokosc_cm:.2f} cm | H: {aktualna_wysokosc_cm:.2f} cm", 10, y_offset)
        y_offset += 20
        
        rysuj_tekst_z_tlem(ramka_przeskalowana, f"Ratio: {stosunek_hw:.3f} | V: {aktualna_objetosc_cm3:.1f} cm3", 10, y_offset)
        y_offset += 20
        rysuj_tekst_z_tlem(ramka_przeskalowana, f"SA: {aktualna_powierzchnia_cm2:.1f} cm2", 10, y_offset)
        y_offset += 20

        rysuj_tekst_z_tlem(ramka_przeskalowana, f"LAB Czerwony: {procent_red_lab:.1f}%", 10, y_offset)
        y_offset += 20
        rysuj_tekst_z_tlem(ramka_przeskalowana, f"LAB Zolty: {procent_yellow_lab:.1f}%", 10, y_offset)
        y_offset += 20
        
        rysuj_tekst_z_tlem(ramka_przeskalowana, f"INNE/USZK: {procent_innych:.1f}%", 10, y_offset)
        y_offset += 20

        rysuj_tekst_z_tlem(ramka_przeskalowana, f"PLC Status: {status_plc}", 10, y_offset)
        
        cv2.imshow('Analiza Ksztaltu Jablka', ramka_przeskalowana)
        
        procenty_do_wyswietlenia = {
            'LAB_JEDNOZNACZNE': aktualne_procenty_kolorow['LAB_JEDNOZNACZNE'],
            'HSV_JEDNOZNACZNE': aktualne_procenty_kolorow['HSV_JEDNOZNACZNE'],
            'BGR_JEDNOZNACZNE': aktualne_procenty_kolorow['BGR_JEDNOZNACZNE']
        }
        grupy_masek['LAB']['GREEN'] = np.zeros(maska_jablka_pojedyncza.shape, dtype=np.uint8) 
        grupy_masek['HSV']['GREEN'] = np.zeros(maska_jablka_pojedyncza.shape, dtype=np.uint8) 
        grupy_masek['BGR']['GREEN'] = np.zeros(maska_jablka_pojedyncza.shape, dtype=np.uint8) 

        wyswietl_montaz_masek(grupy_masek, procenty_do_wyswietlenia, nazwa_wideo, ramka_przeskalowana)
        
        key = cv2.waitKey(CZAS_OCZEKIWANIA_MS) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # --- Wynik Koncowy (Podsumowanie i ZWROT DANYCH) ---
    if not wszystkie_metryki_ksztaltu['ratio']:
        return {"file": sciezka_wideo, "status": "Analiza nie powiodla sie: Brak konturu."}

    srednia_metryka = {k: np.mean(v) for k, v in wszystkie_metryki_ksztaltu.items()}
    srednia_procenty = {f"{k}": np.mean(v) for k, v in wszystkie_procenty.items()} 
    
    srednia_szerokosc_oryg = srednia_metryka['szerokosc']
    srednia_wysokosc_oryg = srednia_metryka['wysokosc']
    
    skala_w = wspolczynniki_skali['W']; skala_h = wspolczynniki_skali['H']
    
    srednia_szerokosc_cm = srednia_szerokosc_oryg * skala_w
    srednia_wysokosc_cm = srednia_wysokosc_oryg * skala_h
    
    skala_powierzchni = skala_w * skala_h
    skala_objetosci = skala_w * skala_w * skala_h
    
    srednia_objetosc_cm3 = srednia_metryka['objetosc'] * skala_objetosci
    srednia_powierzchnia_cm2 = srednia_metryka['powierzchnia'] * skala_powierzchni
    
    srednia_min_wymiar_cm = srednia_metryka['min_wymiar_cm']

    srednia_inna_hsv = 100.0 - (srednia_procenty['red_hsv'] + srednia_procenty['yellow_hsv'])
    srednia_inna_bgr = 100.0 - (srednia_procenty['red_bgr'] + srednia_procenty['yellow_bgr'])
    
    kolor_procenty_wynik = {
        "LAB_JEDNOZNACZNE": {"red_pct": srednia_procenty['red_lab'], "green_pct": 0.0, "yellow_pct": srednia_procenty['yellow_lab'], "other_pct": srednia_procenty['other_lab']},
        "HSV_JEDNOZNACZNE": {"red_pct": srednia_procenty['red_hsv'], "green_pct": 0.0, "yellow_pct": srednia_procenty['yellow_hsv'], "other_pct": srednia_inna_hsv},
        "BGR_JEDNOZNACZNE": {"red_pct": srednia_procenty['red_bgr'], "green_pct": 0.0, "yellow_pct": srednia_procenty['yellow_bgr'], "other_pct": srednia_inna_bgr},
    }
    kategoria, ufnosc = klasyfikuj_jablko(kolor_procenty_wynik, srednia_min_wymiar_cm)
    
    # OBLICZENIE ŚREDNICH WAŻONYCH DLA PLC (LAB: 7, HSV: 2, BGR: 1)
    WAGI = {'LAB': 7.0, 'HSV': 2.0, 'BGR': 1.0}
    SUMA_W = 10.0
    
    def licz_wazona(klucz):
        return (kolor_procenty_wynik['LAB_JEDNOZNACZNE'][klucz] * WAGI['LAB'] + 
                kolor_procenty_wynik['HSV_JEDNOZNACZNE'][klucz] * WAGI['HSV'] + 
                kolor_procenty_wynik['BGR_JEDNOZNACZNE'][klucz] * WAGI['BGR']) / SUMA_W
    
    opis_ksztaltu = "**splaszczone (szerokie)**" if srednia_metryka['ratio'] < 0.95 else \
                         "**wydluzone (wysokie)**" if srednia_metryka['ratio'] > 1.05 else \
                         "**kuliste**"
    
    return {
        "file": sciezka_wideo,
        "ksztalt": {
            "szerokosc_px": srednia_szerokosc_oryg, "wysokosc_px": srednia_wysokosc_oryg,
            "szerokosc_cm": srednia_szerokosc_cm, "wysokosc_cm": srednia_wysokosc_cm,
            "hw_ratio": srednia_metryka['ratio'], "okraglosc": srednia_metryka['powierzchnia'],
            "opis": f"Jablko jest {opis_ksztaltu}",
            "volume_cm_cubed": srednia_objetosc_cm3,
            "surface_area_cm_sq": srednia_powierzchnia_cm2,
            "min_wymiar_cm": srednia_min_wymiar_cm
        },
        "kolor_procenty": kolor_procenty_wynik,
        "jakosc": {
            "avg_other_pct": licz_wazona('other_pct'), # Średnia ważona uszkodzeń
            "avg_red_pct": licz_wazona('red_pct'),     # NOWE: Średnia ważona czerwieni
            "avg_yellow_pct": licz_wazona('yellow_pct')
        },
        "kategoryzacja": {
            "kategoria": kategoria,
            "ufnosc": ufnosc
        },
        "deferred_status": status_odroczonej_wysylki, 
        "numer_elementu": aktualny_numer_elementu 
    }

# --- FUNKCJA DO INICJALIZACJI I URUCHOMIENIA PROGRAMU ---

def pobierz_pliki_wideo_z_folderu(sciezka_folderu, sciezka_kalibracji):
    """Skanuje folder w poszukiwaniu plików wideo (mp4, avi) i pomija plik kalibracyjny."""
    pliki = []
    basename_kalibracji = os.path.basename(sciezka_kalibracji)
    
    for nazwa_pliku in os.listdir(sciezka_folderu):
        pelna_sciezka = os.path.join(sciezka_folderu, nazwa_pliku)
        
        # Sprawdź, czy to plik, czy jest wideo i czy to nie plik kalibracyjny
        if os.path.isfile(pelna_sciezka) and \
           nazwa_pliku.lower().endswith(('.mp4', '.avi', '.mov')) and \
           nazwa_pliku != basename_kalibracji:
            pliki.append(pelna_sciezka)
            
    pliki.sort() 
    return pliki

def inicjalizuj_analize(dane_kalibracji_konfiguracja, sciezka_do_wideo_lub_folderu):
    """Inicjalizuje połączenie PLC i kalibrację przed uruchomieniem Batch Mode."""
    
    wyczysc_terminal() 
    
    # WAŻNE: Upewnij się, że ta ścieżka jest poprawna dla Twojego pliku config.json
    PELNA_SCIEZKA_KONFIGURACJI = r"C:\Users\Jakub Hryniewicz\Documents\studia\moje\sem7\testy\config.json" 
    
    wczytaj_konfiguracje(PELNA_SCIEZKA_KONFIGURACJI)
    
    sciezka_kalibracji = dane_kalibracji_konfiguracja['path']
    WSPOLCZYNNIK_KALIBRACJI_DOMYSLNY = {'W': 0.0125, 'H': 0.0125}
    
    if not os.path.exists(sciezka_kalibracji):
        print(f"OSTRZEŻENIE: Plik kalibracyjny nie istnieje: {sciezka_kalibracji}. Używam domyślnych współczynników.")
        # Zmieniamy nazwę na wspolczynniki_skali, żeby print niżej zadziałał
        wspolczynniki_skali = WSPOLCZYNNIK_KALIBRACJI_DOMYSLNY
    else:
        # Zmieniona funkcja kalibracji (obsługuje ZDJĘCIA)
        wspolczynniki_skali = pobierz_wspolczynnik_kalibracji(
            sciezka_kalibracji, 
            dane_kalibracji_konfiguracja['width_cm'], 
            dane_kalibracji_konfiguracja['height_cm']
        )
    
    # Teraz print zadziała zawsze, bo w obu przypadkach powyżej mamy zmienną 'wspolczynniki_skali'
    print(f"Znormalizowany współczynnik (skala 1.0): W={wspolczynniki_skali['W']:.6f} cm/px, H={wspolczynniki_skali['H']:.6f} cm/px")
    
    # Logika ładowania plików wideo: Sprawdza, czy ścieżka jest folderem czy plikiem
    if os.path.isdir(sciezka_do_wideo_lub_folderu):
        pliki_wideo = pobierz_pliki_wideo_z_folderu(sciezka_do_wideo_lub_folderu, sciezka_kalibracji)
    else:
        # Jeśli podano ścieżkę do jednego pliku (np. jeśli stary plik_wideo był pojedynczą ścieżką)
        pliki_wideo = [sciezka_do_wideo_lub_folderu] 
        if not os.path.exists(sciezka_do_wideo_lub_folderu):
            pliki_wideo = []

    
    klient_plc = snap7.client.Client()
    try:
        klient_plc.connect(IP_PLC, RACK_PLC, SLOT_PLC)
        print(f"NAWIĄZANO POŁĄCZENIE Z PLC: IP={IP_PLC}")

    except Exception as e:
        print(f"BŁĄD POŁĄCZENIA Z PLC ({IP_PLC}): {e}. Kontynuacja bez komunikacji PLC.")
        
    # Uruchomienie trybu Batch bezpośrednio
    przetwarzaj_wideo(wspolczynniki_skali, pliki_wideo, klient_plc)
    
    if klient_plc.get_connected():
        try:
            resetuj_dane_plc_i_lampki(klient_plc)
            klient_plc.disconnect()
        except:
            pass
            
    input("\n--- Analiza Batch zakończona. Naciśnij ENTER, aby wyjść. ---")

# --- FUNKCJA PRZETWARZANIA SERII WIDEO (BATCH MODE) ---

def przetwarzaj_wideo(wspolczynniki_skali, pliki_wideo, klient_plc):
    """
    Analizuje serię dostarczonych plików wideo (Batch Mode).
    """
    
    pliki_wideo_poprawne = [f for f in pliki_wideo if os.path.exists(f)]
        
    if not pliki_wideo_poprawne:
        print("BŁĄD: Brak jakichkolwiek istniejących plików wideo do analizy. Aplikacja zostanie zamknięta.")
        return

    print("\n==============================================")
    print(" 🍎 TRYB BATCH - Analiza serii wideo ")
    print("==============================================\n")

    wynik_odroczony = None
    if klient_plc.get_connected():
        resetuj_dane_plc_i_lampki(klient_plc) 

    wyniki_pomiarow = []
    sumy_kategorii = {}
    
    # Wagi użyte do klasyfikacji
    WAGI = {'LAB': 7.0, 'HSV': 2.0, 'BGR': 1.0}
    SUMA_WAG = 10.0

    for i, sciezka_pliku in enumerate(pliki_wideo_poprawne):
        
        wynik = analizuj_pojedyncze_wideo(
            sciezka_pliku, 
            wspolczynniki_skali, 
            klient_plc, 
            wynik_odroczony, 
            i 
        )
        
        if wynik and wynik.get("status") is None:
            wynik_odroczony = wynik
            wyniki_pomiarow.append(wynik)

            kategoria = wynik["kategoryzacja"]["kategoria"]
            identyfikator_pliku = os.path.basename(wynik["file"]) 
            sumy_kategorii.setdefault(kategoria, []).append(identyfikator_pliku)
        else:
            wynik_odroczony = None
            wyniki_pomiarow.append(wynik) 

    # Krok 3: WYSYŁKA OSTATNIEGO OPÓŹNIONEGO WYNIKU
    if klient_plc.get_connected():
        if wynik_odroczony:
            wyslij_odroczone_dane(klient_plc, wynik_odroczony)
        resetuj_dane_plc_i_lampki(klient_plc)
        
    # Krok 4: Raport Końcowy
    
    print("\n==============================================")
    print(" 📊 RAPORT KOŃCOWY (BATCH MODE) 📊")
    print("==============================================\n")

    suma_przeanalizowanych = len([r for r in wyniki_pomiarow if r and r.get("status") is None])

    if suma_przeanalizowanych > 0:
        
        # --- BLOK 1: SZCZEGÓŁOWE GLOBALNE ŚREDNIE METRYKI ---
        
        if wyniki_pomiarow:
            
            # Pomoce do globalnych obliczeń (dla globalnych statystyk ważonych)
            all_red_wazone = []
            all_yellow_wazone = []
            all_other_wazone = []
            
            srednie_globalne = {
                'szerokosc': np.mean([r['ksztalt']['szerokosc_cm'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'wysokosc': np.mean([r['ksztalt']['wysokosc_cm'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'ratio': np.mean([r['ksztalt']['hw_ratio'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'objetosc': np.mean([r['ksztalt']['volume_cm_cubed'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'powierzchnia': np.mean([r['ksztalt']['surface_area_cm_sq'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'other_lab': np.mean([r['kolor_procenty']['LAB_JEDNOZNACZNE']['other_pct'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'other_hsv': np.mean([r['kolor_procenty']['HSV_JEDNOZNACZNE']['other_pct'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'other_bgr': np.mean([r['kolor_procenty']['BGR_JEDNOZNACZNE']['other_pct'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'red_lab': np.mean([r['kolor_procenty']['LAB_JEDNOZNACZNE']['red_pct'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'yellow_lab': np.mean([r['kolor_procenty']['LAB_JEDNOZNACZNE']['yellow_pct'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'red_hsv': np.mean([r['kolor_procenty']['HSV_JEDNOZNACZNE']['red_pct'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'yellow_hsv': np.mean([r['kolor_procenty']['HSV_JEDNOZNACZNE']['yellow_pct'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'red_bgr': np.mean([r['kolor_procenty']['BGR_JEDNOZNACZNE']['red_pct'] for r in wyniki_pomiarow if r and r.get("status") is None]),
                'yellow_bgr': np.mean([r['kolor_procenty']['BGR_JEDNOZNACZNE']['yellow_pct'] for r in wyniki_pomiarow if r and r.get("status") is None]),
            }
            
            for r in wyniki_pomiarow:
                if r and r.get("status") is None:
                    kp = r['kolor_procenty']
                    red_wazone = ((kp['LAB_JEDNOZNACZNE']['red_pct'] * WAGI['LAB']) + (kp['HSV_JEDNOZNACZNE']['red_pct'] * WAGI['HSV']) + (kp['BGR_JEDNOZNACZNE']['red_pct'] * WAGI['BGR'])) / SUMA_WAG
                    yellow_wazone = ((kp['LAB_JEDNOZNACZNE']['yellow_pct'] * WAGI['LAB']) + (kp['HSV_JEDNOZNACZNE']['yellow_pct'] * WAGI['HSV']) + (kp['BGR_JEDNOZNACZNE']['yellow_pct'] * WAGI['BGR'])) / SUMA_WAG
                    other_wazone = ((kp['LAB_JEDNOZNACZNE']['other_pct'] * WAGI['LAB']) + (kp['HSV_JEDNOZNACZNE']['other_pct'] * WAGI['HSV']) + (kp['BGR_JEDNOZNACZNE']['other_pct'] * WAGI['BGR'])) / SUMA_WAG
                    all_red_wazone.append(red_wazone)
                    all_yellow_wazone.append(yellow_wazone)
                    all_other_wazone.append(other_wazone)

            # Globalny Opis Kształtu
            srednie_ratio_globalne = srednie_globalne['ratio']
            opis_ksztaltu_globalny = "**splaszczone (szerokie)**" if srednie_ratio_globalne < 0.95 else \
                                     "**wydluzone (wysokie)**" if srednie_ratio_globalne > 1.05 else \
                                     "**kuliste**"

            print("--- 🔬 SZCZEGÓŁOWE GLOBALNE ŚREDNIE METRYKI ---")
            
            print(f"  Średnia Szerokość: {srednie_globalne['szerokosc']:.2f} cm")
            print(f"  Średnia Wysokość: {srednie_globalne['wysokosc']:.2f} cm")
            print(f"  Średni Stosunek H/W: {srednie_globalne['ratio']:.3f} ({opis_ksztaltu_globalny})")
            print(f"  Średnia Objętość: {srednie_globalne['objetosc']:.1f} cm³")
            print(f"  Średnia Powierzchnia: {srednie_globalne['powierzchnia']:.1f} cm²")
            print("-" * 50)
            
            print("  Globalna Średnia Uszkodzeń/Innych:")
            print(f"  > WAŻONA ({WAGI['LAB']}/{WAGI['HSV']}/{WAGI['BGR']}): {np.mean(all_other_wazone):.1f} %")
            print(f"  > LAB (Wartość): {srednie_globalne['other_lab']:.1f} %")
            print("-" * 50)
            
            print("  Globalne Udziały Kolorów (WAŻONE - RED/YELLOW):")
            
            print(f"  > Czerwony (WAŻONA): {np.mean(all_red_wazone):.1f} %")
            print(f"  > Żółty (WAŻONA): {np.mean(all_yellow_wazone):.1f} %")
            print("-" * 50)
            print(f"  > Czerwony (lab): {srednie_globalne['red_lab']:.1f} %")
            print(f"  > Żółty (lab): {srednie_globalne['yellow_lab']:.1f} %")
            print("-" * 50)
            print(f"  > Czerwony (hsv): {srednie_globalne['red_hsv']:.1f} %")
            print(f"  > Żółty (hsv): {srednie_globalne['yellow_hsv']:.1f} %")
            print("-" * 50)
            print(f"  > Czerwony (rgb): {srednie_globalne['red_bgr']:.1f} %")
            print(f"  > Żółty (rgb): {srednie_globalne['yellow_bgr']:.1f} %")
            print("-" * 50)
        # --- BLOK 2: INDYWIDUALNE PODSUMOWANIE ANALIZY (ROZSZERZONE) ---
        
        print("\n--- INDYWIDUALNE PODSUMOWANIE ANALIZY ---")
        for wynik in wyniki_pomiarow:
            if wynik and wynik.get("status") is None:
                kp = wynik['kolor_procenty']
                ksztalt = wynik['ksztalt'] 
                kategoria_koncowa = wynik['kategoryzacja']['kategoria']
                
                # Obliczanie uśrednionych procentów koloru i uszkodzeń (WAŻONE)
                srednia_red_total = ((kp['LAB_JEDNOZNACZNE']['red_pct'] * WAGI['LAB']) + (kp['HSV_JEDNOZNACZNE']['red_pct'] * WAGI['HSV']) + (kp['BGR_JEDNOZNACZNE']['red_pct'] * WAGI['BGR'])) / SUMA_WAG
                srednia_yellow_total = ((kp['LAB_JEDNOZNACZNE']['yellow_pct'] * WAGI['LAB']) + (kp['HSV_JEDNOZNACZNE']['yellow_pct'] * WAGI['HSV']) + (kp['BGR_JEDNOZNACZNE']['yellow_pct'] * WAGI['BGR'])) / SUMA_WAG
                srednia_innych_total = ((kp['LAB_JEDNOZNACZNE']['other_pct'] * WAGI['LAB']) + (kp['HSV_JEDNOZNACZNE']['other_pct'] * WAGI['HSV']) + (kp['BGR_JEDNOZNACZNE']['other_pct'] * WAGI['BGR'])) / SUMA_WAG


                print(f"--- PLIK: {os.path.basename(wynik['file'])} (Nr: {wynik['numer_elementu']}) ---")
                print(f"  Kategoria: **{kategoria_koncowa}** | V: {ksztalt['volume_cm_cubed']:.1f} cm³")
                print(f"  Wymiary: {ksztalt['szerokosc_cm']:.2f} cm (W) x {ksztalt['wysokosc_cm']:.2f} cm (H) | Min. Wymiar: {ksztalt['min_wymiar_cm']:.2f} cm")
                print(f"  Kształt: {ksztalt['opis']} | Ratio: {ksztalt['hw_ratio']:.3f}")
                print(f"  Czerwony (Średnia WAŻONA): {srednia_red_total:.1f}%")
                print(f"  Czerwony (Detale): LAB: {kp['LAB_JEDNOZNACZNE']['red_pct']:.1f}% | HSV: {kp['HSV_JEDNOZNACZNE']['red_pct']:.1f}% | BGR: {kp['BGR_JEDNOZNACZNE']['red_pct']:.1f}%")
                print(f"  Zolty (Średnia WAŻONA): {srednia_yellow_total:.1f}%")
                print(f"  Zolty (Detale): LAB: {kp['LAB_JEDNOZNACZNE']['yellow_pct']:.1f}% | HSV: {kp['HSV_JEDNOZNACZNE']['yellow_pct']:.1f}% | BGR: {kp['BGR_JEDNOZNACZNE']['yellow_pct']:.1f}%")
                print(f"  Uszkodzenia (Średnia WAŻONA): {srednia_innych_total:.1f}%")
                print(f"  Uszkodzenia (Detale): LAB: {kp['LAB_JEDNOZNACZNE']['other_pct']:.1f}% | HSV: {kp['HSV_JEDNOZNACZNE']['other_pct']:.1f}% | BGR: {kp['BGR_JEDNOZNACZNE']['other_pct']:.1f}%")
                print(f"  % KOLORU (Średnia WAŻONA): Cz: {srednia_red_total:.1f}%, Żł: {srednia_yellow_total:.1f}%")
                print("-" * 50)

        # --- BLOK 3: PODSUMOWANIE KATEGORYZACJI ---
        
        print("\n--- GLOBALNE PODSUMOWANIE KATEGORYZACJI ---")

        uporzadkowane_kategorie = ['RED_KLASA1', 'RED_KLASA2', 'RED_KLASA3', 'YELLOW_KLASA1', 'YELLOW_KLASA2', 'YELLOW_KLASA3', 'USZKODZONE', 'ODRZUT']
        for kategoria in uporzadkowane_kategorie:
            identyfikatory = sumy_kategorii.get(kategoria, [])
            liczba = len(identyfikatory)
            if liczba > 0:
                procent = (liczba / suma_przeanalizowanych) * 100
                print(f"[{kategoria}]: **{liczba}** sztuk ({procent:.1f}%)")
        print(f"\nŁącznie przeanalizowano: **{suma_przeanalizowanych}** sztuk.")

    else:
        print("Nie przeanalizowano żadnego pliku wideo. Sprawdź ścieżki dostępu i logi błędów.")


# --- KALIBRACJA ORAZ UZYCIE PROGRAMU ---

# 1. DANE KALIBRACYJNE (Wymaga ISTNIEJĄCEGO ZDJĘCIA z przedmiotem o znanych wymiarach)
dane_kalibracyjne = {
    # Zmień ścieżkę na TWÓJ plik JPG/PNG (teraz funkcja obsługuje zdjęcia!)
    'path': r"C:\Users\Jakub Hryniewicz\Documents\studia\moje\sem7\testy\\kalibracja\kalibracja4.jpg", 
    'width_cm': 5.7,   # PRZYKŁADOWA SZEROKOŚĆ (wymiary muszą pasować do fizycznego obiektu!)
    'height_cm': 12.9, # PRZYKŁADOWA WYSOKOŚĆ (wymiary muszą pasować do fizycznego obiektu!)
}


# 2. FOLDER ZAWIERAJĄCY PLIKI WIDEO DO ANALIZY
# Wystarczy podać ścieżkę do folderu, a kod sam znajdzie pliki .mp4, .avi, .mov.
FOLDER_Z_WIDEO_DO_ANALIZY = r"C:\Users\Jakub Hryniewicz\Documents\studia\moje\sem7\testy\jablka3"


# Rozpoczęcie przetwarzania
if __name__ == '__main__':
    # Używamy ścieżki do folderu dla trybu Batch
    inicjalizuj_analize(dane_kalibracyjne, FOLDER_Z_WIDEO_DO_ANALIZY)