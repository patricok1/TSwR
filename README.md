# Implementacja i analiza algorytmu sterowania MPPI dla symulowanego samochodu

Projekt realizowany w ramach przedmiotu TSwR.
Celem projektu jest implementacja zaawansowanego środowiska symulacyjnego dla autonomicznego samochodu wyścigowego oraz wykorzystanie algorytmu MPPI (Model Predictive Path Integral) do zadania śledzenia optymalnej trajektorii.

# Opis projektu
Projekt polega na stworzeniu symulatora pojazdu opartego na dynamicznym modelu "rowerowym" (bicycle model) oraz implementacji stochastycznego kontrolera predykcyjnego. W odróżnieniu od klasycznych metod optymalizacji (jak NMPC), MPPI wykorzystuje masywne równoległe próbkowanie trajektorii, co pozwala na efektywne sterowanie w nieliniowych i nietrywialnych reżimach jazdy.  
Głównym zadaniem jest utrzymanie pojazdu na zadanej ścieżce przy jednoczesnym uwzględnieniu ograniczeń fizycznych, takich jak elipsa tarcia i granice toru.  

# Model Matematyczny (Fundament)
Wytyczne:
• Układ współrzędnych: Współrzędne krzywoliniowe (curvilinear coordinates) względem ścieżki referencyjnej. 
• Wektor stanu
• Ograniczenia: Implementacja elipsy tarcia (friction ellipse) ograniczającej sumaryczne siły działające na koła.

# Stos technologiczny
• PyTorch: Wykorzystanie tensorów do równoległego obliczania tysięcy próbek trajektorii na GPU/CPU.
• pytorch-mppi: Biblioteka implementująca logikę algorytmu Model Predictive Path Integral.
• Numpy / Scipy: Wsparcie dla obliczeń macierzowych i parametrów modelu.
• Matplotlib: Wizualizacja toru, pojazdu oraz „chmury” próbek generowanych przez MPPI.

# Cel projektu
1.	Implementacja środowiska: Stworzenie modelu dynamiki pojazdu w oparciu o równania różniczkowe z artykułu.  
2.	Przygotowanie toru: Definicja trajektorii referencyjnej (np. linia środkowa toru wyścigowego).
3.	Integracja MPPI: Skonfigurowanie funkcji kosztu (cost function) promującej postęp wzdłuż trasy i karzącej za wypadnięcie z toru.  
4.	Analiza wydajności: Porównanie wpływu liczby generowanych próbek (samples) na precyzję śledzenia i czas odpowiedzi sterownika.
5.	Testy graniczne: Badanie stabilności pojazdu przy dużych przyspieszeniach bocznych (at-limit handling).

# Cel na połowę projektu
• Działający model matematyczny (bicycle model) zaimplementowany w Pythonie.
• Prosta wizualizacja toru i pozycji pojazdu.
• Uruchomienie podstawowej pętli sterowania MPPI pozwalającej na przejechanie prostego odcinka trasy.
# Planowany rezultat końcowy

• W pełni funkcjonalne środowisko symulacyjne.
• Kontroler MPPI zdolny do stabilnego prowadzenia auta po złożonym torze wyścigowym.
• Zestaw eksperymentów pokazujących przewagi MPPI w radzeniu sobie z nieliniowościami modelu opon.
• Analiza kosztu obliczeniowego i porównanie wyników z założeniami teoretycznymi z artykułu.
# Bibliografia
1. https://arxiv.org/pdf/2003.04882.pdf
2.	pytorch-mppi library documentation
