# Implementacja i analiza algorytmu sterowania MPPI dla symulowanego samochodu

Projekt realizowany w ramach przedmiotu TSwR.
Celem projektu jest implementacja zaawansowanego środowiska symulacyjnego dla autonomicznego samochodu wyścigowego oraz wykorzystanie algorytmu MPPI (Model Predictive Path Integral) do zadania śledzenia optymalnej trajektorii.

# Opis projektu
Projekt polega na stworzeniu symulatora pojazdu opartego na dynamicznym modelu "rowerowym" (bicycle model) oraz implementacji stochastycznego kontrolera predykcyjnego. W odróżnieniu od klasycznych metod optymalizacji (jak NMPC), MPPI wykorzystuje masywne równoległe próbkowanie trajektorii, co pozwala na efektywne sterowanie w nieliniowych i nietrywialnych reżimach jazdy.  
Głównym zadaniem jest utrzymanie pojazdu na zadanej ścieżce przy jednoczesnym uwzględnieniu ograniczeń fizycznych, takich jak elipsa tarcia i granice toru.  

# Model Matematyczny (Fundament)
Wytyczne:
1. Układ współrzędnych: Współrzędne krzywoliniowe (curvilinear coordinates) względem ścieżki referencyjnej.
 
2. Wektor stanu
3. Ograniczenia: Implementacja elipsy tarcia (friction ellipse) ograniczającej sumaryczne siły działające na koła.

# Stos technologiczny
1. PyTorch: Wykorzystanie tensorów do równoległego obliczania tysięcy próbek trajektorii na GPU/CPU.
2. pytorch-mppi: Biblioteka implementująca logikę algorytmu Model Predictive Path Integral.
3. Numpy / Scipy: Wsparcie dla obliczeń macierzowych i parametrów modelu.
4. Matplotlib: Wizualizacja toru, pojazdu oraz „chmury” próbek generowanych przez MPPI.

# Cel projektu
1.	Implementacja środowiska: Stworzenie modelu dynamiki pojazdu w oparciu o równania różniczkowe z artykułu.  
2.	Przygotowanie toru: Definicja trajektorii referencyjnej (np. linia środkowa toru wyścigowego).
3.	Integracja MPPI: Skonfigurowanie funkcji kosztu (cost function) promującej postęp wzdłuż trasy i karzącej za wypadnięcie z toru.  
4.	Analiza wydajności: Porównanie wpływu liczby generowanych próbek (samples) na precyzję śledzenia i czas odpowiedzi sterownika.
5.	Testy graniczne: Badanie stabilności pojazdu przy dużych przyspieszeniach bocznych (at-limit handling).

# Cel na połowę projektu
1. Działający model matematyczny (bicycle model) zaimplementowany w Pythonie.
2. Prosta wizualizacja toru i pozycji pojazdu.
3. Uruchomienie podstawowej pętli sterowania MPPI pozwalającej na przejechanie prostego odcinka trasy.
# Planowany rezultat końcowy

1. W pełni funkcjonalne środowisko symulacyjne.
2. Kontroler MPPI zdolny do stabilnego prowadzenia auta po złożonym torze wyścigowym.
3. Zestaw eksperymentów pokazujących przewagi MPPI w radzeniu sobie z nieliniowościami modelu opon.
4. Analiza kosztu obliczeniowego i porównanie wyników z założeniami teoretycznymi z artykułu.
# Bibliografia
1. https://arxiv.org/pdf/2003.04882.pdf
2.	pytorch-mppi library documentation
