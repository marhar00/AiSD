"""
Algorytmy i Struktury Danych - Projekt 2: Wyprawa w Alpy


Problem: Wybór k szczytów alpejskich spełniających preferencje profesora Hilary'ego
"""

from typing import List, Tuple
import random
import time


class Peak:
    """Reprezentacja szczytu górskiego"""
    def __init__(self, id: int, height: int, time: int):
        self.id = id
        self.height = height
        self.time = time

    def __repr__(self):
        return f"Peak({self.id}, height={self.height}, time={self.time})"

    def __lt__(self, other):
        """Porównanie według preferencji profesora"""
        if self.height != other.height:
            return self.height < other.height  # Wyższy jest lepszy
        return self.time > other.time  # Przy tej samej wysokości, krótszy czas jest lepszy
    
    @classmethod
    def from_tuple(cls, t: tuple) -> "Peak":
        return cls(*t)

def Partition(peaks: List[Peak], l: int, r:int, method:str) -> int:
    """ Tutaj został zaimplementowany algorytm Partition """
    """ Pivot zostanie wybrany losowo, w śrendim przypadku złozoność ma być liniowa """
    """ Po tym algorytmie lista będzie przekształcona"""
    assert r <= len(peaks) - 1

    if method == "random":
        x_indx = random.randint(l,r)
    if method == "first":
        x_indx = l 
    x = peaks[x_indx] 
    # W schemacie partition na wykładzie x czyli nasz pivot jest na samym początku ciągu, więc chce przenieść nasz pivot na początek
    peaks[x_indx], peaks[l] = peaks[l], peaks[x_indx]
    i, j = l, r+1
    while True: 
        i += 1
        while peaks[i] < x and i<r: 
            i+= 1 
        j -= 1
        while peaks[j] > x and j>l:
            j-= 1
        if i < j:
            peaks[i], peaks[j] = peaks[j], peaks[i]     
        else:
            break
    # dajemy pivota na prawdziwe miejsce 
    peaks[l], peaks[j] = peaks[j], peaks[l]
    return j 

def alpy_algorithm(peaks: List[Peak], k:int, G:int) -> Tuple[list, List[Peak]]:
    
    peaks_local = peaks.copy()

    k0 = k

    peaks_local = [p for p in peaks_local if peaks_local.time <= G]

    l = 0
    r = len(peaks_local) - 1
    if l >= r:
        # nothing to partition; return up to k ids
        return [p.id for p in peaks_local[:k]], peaks_local

    j = None
    while l < r:
        j = Partition(peaks_local, l, r, "first")
        if j == r - k + 1:
            break
        if j < r - k + 1:
            l = j + 1
        else:
            k = k - (r - j + 1)
            r = j - 1

    result = [elem.id for elem in peaks_local[len(peaks_local) - k0:]] 
    return result, peaks_local

class AlpineExpedition:
    """Klasa implementująca algorytm wyboru szczytów alpejskich."""

    def __init__(self, n: int, k: int, G: int, peaks: List[Peak]):
        """
        Inicjalizacja problemu wyboru szczytów.

        Args:
            n: liczba dostępnych szczytów 
            k: liczba szczytów do wybrania
            G: maksymalny czas na wyprawę (w godzinach)
            peaks: lista krotek (wysokość, czas) dla każdego szczytu
        """
        self.n = n
        self.k = k
        self.G = G
        self.peaks = peaks
        self.selected_indices = []
        self.execution_time = 0
        self.comparisons = 0

    def Partition(self, l:int, r:int) -> int:
        """ Tutaj został zaimplementowany algorytm Partition """
        """ Pivot zostanie wybrany losowo, w śrendim przypadku złozoność ma być liniowa """
        """ Po tym algorytmie lista będzie przekształcona"""

        assert r <= len(self.peaks) - 1

        x_indx = l 
        x = self.peaks[x_indx] 
        # W schemacie partition na wykładzie x czyli nasz pivot jest na samym początku ciągu, więc chce przenieść nasz pivot na początek
        self.peaks[x_indx], self.peaks[l] = self.peaks[l], self.peaks[x_indx]
        i, j = l, r+1
        while True: 
            i += 1
            while i<r: 
                self.comparisons += 1
                if self.peaks[i] < x:
                    i+= 1 
            j -= 1
            while j>l:
                self.comparisons += 1
                if self.peaks[j] > x:
                    j-= 1
            if i < j:
                self.peaks[i], self.peaks[j] = self.peaks[j], self.peaks[i]     
            else:
                break
        # dajemy pivota na prawdziwe miejsce 
        self.peaks[l], self.peaks[j] = self.peaks[j], self.peaks[l]
        return j 

    def alpy_algorithm(self) -> List[int]:
        peaks_local = self.peaks.copy()
        start_time = time.perf_counter()
        k0 = self.k
        i = 0
        while i < len(peaks_local):
            self.comparisons +=1 
            if peaks_local[i].time > self.G:
                del peaks_local[i]
            else:
                i += 1

        l = 0
        r = len(peaks_local) - 1
        if l >= r: 
            # nothing to partition; return up to k ids
            return [p.id for p in peaks_local[:self.k]]

        j = None
        while l < r:
            j = Partition(peaks_local,l,r, "first")
            if j == r - self.k + 1:
                break
            if j < r - self.k + 1:
                l = j + 1
            else:
                self.k = self.k - (r - j + 1)
                r = j - 1

        result = [elem.id for elem in peaks_local[len(peaks_local) - k0:]] 
        self.selected_indices = result
        self.execution_time = time.perf_counter() - start_time
        return result

    def get_statistics(self) -> dict:

        return {
            'n': self.n,
            'k': self.k,
            'G': self.G,
            'selected_count': len(self.selected_indices),
            'execution_time': self.execution_time,
            'comparisons': self.comparisons,
            'selected_indices': self.selected_indices
        }

    def get_selected_peaks_info(self) -> List[Tuple[int, int, int]]:
        """
        Zwraca informacje o wybranych szczytach.

        Returns:
            Lista krotek (indeks, wysokość, czas)
        """
        result = []
        for idx in self.selected_indices:
            # peaks are stored as Peak instances; indices are 1-based in this project
            peak = self.peaks[idx - 1]
            result.append((idx, peak.height, peak.time))
        return result

    def generate_random_peaks(n: int,
                          height_range: Tuple[int, int] = (1000, 5000),
                          time_range: Tuple[int, int] = (2, 24)) -> List[Peak]:
        """
        Generuje losowy zestaw szczytów.

        Args:
            n: liczba szczytów
            height_range: zakres wysokości (min, max)
            time_range: zakres czasu (min, max)

        Returns:
            Lista krotek (wysokość, czas)
        """
        peaks: List[Peak] = []
        for i in range(1, n + 1):
            height = random.randint(height_range[0], height_range[1])
            duration = random.randint(time_range[0], time_range[1])
            peaks.append(Peak(i, height, duration))
        return peaks

    def generate_worst_case_increasing(n: int) -> List[Peak]:
        return [Peak(i, i * 10, i) for i in range(1, n + 1)]

    def generate_worst_case_decreasing(n: int) -> List[Peak]:
        return [Peak(i, i * 10, i) for i in range(n, 0, -1)]

    def generate_best_case_peaks(n: int, k:int) -> List[Peak]:
        """
        Generuje optymistyczny przypadek - na pierwszym miejscu znajduje się szczyt k-tej wielkości

        Args:
            n: liczba szczytów

        Returns:
            Lista krotek (wysokość, czas) rosnące
        """

        peaks = AlpineExpedition.generate_random_peaks(n)
        # teraz szukamy k-tego elementu
        peaks_sorted = sorted(peaks)
        idx = n - k + 1 # miejsce k-tej wielkości elementu w liście posortowanje rosnąco
        pivot = peaks_sorted[idx]
        indx = peaks.index(pivot)
        peaks[0], peaks[indx] = pivot, peaks[0]
        peaks[0], peaks[indx] = pivot, peaks[0]

        return peaks
        

    def read_input_from_keyboard() -> Tuple[int, int, int, List[Peak]]:
        """
        Wczytuje dane wejściowe z klawiatury

        Returns:
            Krotka (n, k, G, peaks)

        """
        print("=== Wyprawa w Alpy - System wyboru szczytów ===\n")

        n = int(input("Podaj liczbę szczytów wokół kurortu (n): "))
        k = int(input("Podaj liczbę planowanych wypraw (k): "))
        G = int(input("Podaj maksymalny czas na wyprawę w godzinach (G): "))

        print(f"\nPodaj charakterystyki {n} szczytów (wysokość czas):")
        peaks: List[Peak] = []
        for i in range(n):
            line = input(f"Szczyt {i+1}: ")
            height, duration = map(int, line.split())
            peaks.append(Peak(i + 1, height, duration))

        return n, k, G, peaks

    def is_correct(peaks: List[Peak], k:int, G:int):  
        """ Ta funkcja sprawdza czy rozwiązanie przez algorytm jest poprawne """
        peaks_selected = []
        for i in range(len(peaks)):
            if peaks[i].time <= G:
                peaks_selected.append(peaks[i])
        peaks_sorted = [elem.id for elem in sorted(peaks_selected, reverse=True)[:k]]
        peaks_local = peaks.copy()
        result, _ = alpy_algorithm(peaks_local, k,G)
        # bierzemy do k-tego elemntu włącznie 
        return sorted(peaks_sorted) == sorted(result), result, peaks_sorted
        # najpierw bool, czy jest dobrze czy nie, potem indeksy z algorytmu, potem indeksy z posortowanej listy, a na końcu listę po algorytmie, a potem lista, gdzie na k-tym miejscu powinien znajdować się k-ty element

    def is_k(peaks: List[Peak], k:int) -> bool:
        """ Tutaj mamy sprawdzanie czy algorytm działa prawdidłowo"""
        n = len(peaks)
        _, result= AlpineExpedition.alpy_algorithm(peaks,k,25)
        sort_result = sorted(peaks)
        if sort_result[n-k] != result[n-k]:
            print("Na złym miejscu jest pivot")
            return False
        pivot = result[n-k]
        for i in range(n-k):
            if result[i] > pivot:
                print("Po lewej sa większe")
                return False  
        for i in range(n-k+1, n):
            if result[i] < pivot:
                print("Po prawej są większe")
                return False 
        return True

    def length_correct(peaks : List[Peak], k:int) -> int:
        result, _ = AlpineExpedition.alpy_algorithm(peaks, k, 25)
        return len(result)

    def has_duplicates(peaks: List[Peak]) -> bool:
        seen = set()
        for elem in peaks:
            peak_tuple = (elem.height, elem.time )
            if peak_tuple in seen:
                return True
            else:
                seen.add(peak_tuple)
        return False

def main():
    """Funkcja główna programu."""
    print("")
    print("=== Algorytmy i Struktury Danych ===")
    print("")
    print("=== Projekt 2: Wyprawa w Alpy ===\n")

    # Przykład 1: Dane z klawiatury
    print("Wybierz tryb:")
    print("")
    print("1 - Wczytaj dane z klawiatury")
    print("2 - Generuj losowe dane")
    print("")
    choice = input("Twój wybór: ")

    if choice == "1":
        n, k, G, peaks = AlpineExpedition.read_input_from_keyboard()
        bool1,result_algorithm, result_sorted = AlpineExpedition.is_correct(peaks, k, G)
        if bool1:
            print("")
            print("Algorytm zadziałał poprawnie!!!")
            print("")
            print(f"Wybrane indeksy szczytów: {result_algorithm}")
        else:
            print("")
            print("Algorytm nie zadziałał poprawnie")
    elif choice == "2":
        n = int(input("Podaj liczbę szczytów (n): "))
        k = int(input("Podaj liczbę do wybrania (k): "))
        G = int(input("Podaj limit czasowy (G): "))
        peaks = AlpineExpedition.generate_random_peaks(n)
        print("")
        print("Zostały wylosowane natsępujące dane:")
        print("")
        print(peaks)
        print("")
        bool1,result_algorithm, result_sorted = AlpineExpedition.is_correct(peaks, k,G)
        print("")
        print("--- Czy rozwiązanie jest poprawne? ---")
        print("")
        if bool1:
            print("Algorytm zadziałał poprawnie!!!")
            print("")
            print(f"Wynik indeksów z algorytmu: {result_algorithm}")
            print("")
            print(f"Wynik z posorotowania: {result_sorted}")
        else:
            print("Algorytm nie zadziałał poprawnie ;(")
    else:
        print("Został wybrany zły numer")

    
if __name__ == "__main__":
    main()
