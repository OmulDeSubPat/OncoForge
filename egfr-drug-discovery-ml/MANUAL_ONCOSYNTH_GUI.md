# Manual de utilizare pentru GUI-ul OncoSynth

## 1. Scopul aplicatiei

`OncoSynth` este o interfata grafica pentru monitorizarea, analiza si trierea moleculelor generate in cadrul platformei AI de drug discovery orientate pe tinta `EGFR`.

Interfata este conceputa pentru:

- controlul sesiunilor de generare
- evaluarea moleculelor selectate
- comparatia cu molecule de referinta din piata
- urmarirea contributiei agentilor AI si a indicatorilor RLVR
- gestionarea fluxului de laborator prin marcare, revizie si export

## 2. Pornirea aplicatiei

Aplicatia functioneaza in doi pasi: backend si frontend.

### Backend

Din radacina proiectului:

```powershell
cd "D:\ONCS 2026\egfr-drug-discovery-ml"
python -m uvicorn src.gui.oncoforge_api.app:app --host 127.0.0.1 --port 8000
```

### Frontend

Intr-un terminal separat:

```powershell
cd "D:\ONCS 2026\egfr-drug-discovery-ml\apps\oncoforge-ui"
npm run dev
```

Aplicatia este accesibila la:

```text
http://127.0.0.1:5173
```

## 3. Moduri de utilizare

Interfata ofera doua moduri principale:

### Mod simplificat

Mod orientat pe decizie rapida. Este recomandat pentru:

- evaluarea moleculei active
- comparatie rapida cu candidatii de top
- consultarea verdictului si a indicatorilor principali
- selectie operationala din biblioteca

### Mod avansat

Mod orientat pe analiza avansata. Este recomandat pentru:

- audit RLVR
- analiza cronologiei iterative
- planificare experimentala
- export si jurnal operational

## 4. Structura generala a interfetei

### Bara superioara

Contine:

- numele sesiunii curente
- statusul sesiunii
- modul de generare activ
- modul UI selectat
- comenzile principale: `Porneste generarea`, `Stop generare`, `Actualizeaza`, `Reseteaza`, `Import`, `Export`

Tot aici sunt afisate:

- starea sesiunii
- progresul curent
- ultima actualizare
- numarul de molecule din setul de comparatie
- numarul de molecule aflate in revizie

### Rezumat executiv pentru molecula activa

Aceasta zona este conceputa pentru primul contact cu sesiunea si trebuie citita prima.

Include:

- molecula activa
- verdictul rapid
- actiunea recomandata
- top candidati
- doua grafice live pentru scor si promovari

### Navigatie

Navigatia este compacta si permite acces rapid intre sectiuni. In modul `simplificat`, sunt afisate doar sectiunile esentiale. In modul `avansat`, sunt disponibile si zonele de audit, planificare, export si jurnal.

### Cautare unificata

Panoul de cautare permite identificarea rapida a:

- unei sectiuni
- unei molecule
- unei rute sintetice
- unui comparator de piata

Panoul este colapsabil si poate fi extins doar atunci cand este necesar.

### Setul de comparatie

Acest panou retine 2-4 molecule selectate pentru analiza comparativa. Setul ramane disponibil intre sectiuni si poate fi deschis direct in ecranul de comparatie.

Panoul este colapsabil si ramane restrans automat atunci cand nu contine molecule.

## 5. Fluxul standard de lucru

Ordinea recomandata de utilizare este urmatoarea:

1. Porniti sesiunea de generare.
2. Verificati rezumatul executiv si candidatii de top.
3. Deschideti molecula activa pentru analiza detaliata.
4. Transferati molecule relevante in setul de comparatie.
5. Consultati sectiunea de risc si fundamentarea prioritizarii.
6. Aplicati marcajele de laborator: `Fixeaza`, `Aproba`, `Respinge`, `Retesteaza`.
7. Exportati lista prioritara aprobata.

## 6. Sectiuni principale

### Sesiune

Afiseaza:

- parametrii curenti ai sesiunii
- sumarul rezultatelor
- indicatori globali ai lotului curent
- grafice de evolutie pe runde

Utilizati aceasta sectiune pentru confirmarea starii generale a sesiunii.

### Triere

Afiseaza:

- reponderare interactiva
- heatmap pe generatii
- grafice de stabilitate si incredere

Utilizati aceasta sectiune pentru a intelege cum se modifica ranking-ul in functie de ponderi si criterii.

### Molecula

Aceasta este sectiunea centrala pentru analiza decizionala.

Include:

- vizualizare 2D si 3D
- scoruri cheie
- comparator de piata
- semnale pro si contra
- praguri trecute si picate
- contributia agentilor AI

### Comparatie

Permite:

- comparatia moleculei active cu `Osimertinib`, `Gefitinib`, `Erlotinib`
- analiza paralela pentru 2-4 molecule
- consultarea deltelor fata de candidatul activ

### Risc si explicatii

Permite:

- consultarea argumentelor pro si contra
- evaluarea euristica ADMET / semnale de risc
- revizuirea istoricului de decizie
- completarea carnetului chimistului

### Biblioteca

Contine:

- grafice pentru distributia moleculelor
- o lista paginata si filtrabila
- selectie rapida a moleculelor
- transfer direct in setul de comparatie

### Audit IA

Disponibila in modul `avansat`.

Contine:

- evolutia recompensei verificabile
- penalizari RLVR
- cronologie iterativa
- fluxuri intre agenti

### Planificare

Disponibila in modul `avansat`.

Contine:

- plan experimental
- comparatie intre sesiuni
- filtrare a notelor din carnet

### Export

Disponibila in modul `avansat`.

Permite:

- exportul moleculei selectate
- exportul moleculelor promovate
- exportul setului marcat pentru revizie
- import partial din JSON sau CSV
- restaurarea draftului local

### Jurnal

Disponibila in modul `avansat`.

Contine:

- loguri operationale
- surse utilizate
- mesaje utile pentru audit si depanare

## 7. Operatiuni esentiale

### Selectarea unei molecule

O molecula poate fi selectata din:

- rezumatul executiv
- biblioteca
- rezultatele cautarii
- comparatie multipla

La selectie, aplicatia deschide automat ecranul dedicat moleculei.

### Adaugarea in setul de comparatie

O molecula poate fi adaugata in setul de comparatie prin butoanele `Compara` sau `Adauga`.

Setul de comparatie este:

- persistent intre sectiuni
- limitat la 4 molecule
- disponibil imediat in sectiunea `Comparatie`

### Marcaje de laborator

Sectiunea `Risc si explicatii` permite marcarea unei molecule cu urmatoarele actiuni:

- `Fixeaza`
- `Aproba`
- `Respinge`
- `Retesteaza`

Aceste marcaje sunt salvate local si pot fi folosite ulterior la filtrare si export.

## 8. Exportul listei prioritare

In sectiunea `Export`, selectati domeniul de export `Revizie` pentru a extrage moleculele marcate in carnetul chimistului cu actiuni de laborator relevante.

Formatele disponibile sunt:

- `JSON`
- `CSV`

## 9. Salvare automata si restaurare

Aplicatia poate salva local:

- parametrii sesiunii
- sectiunea activa
- molecula selectata
- setul de comparatie
- starea panourilor colapsabile

Restaurarea se face din sectiunea `Export`.

## 10. Recomandari de utilizare

- Utilizati modul `simplificat` pentru triere si selectie rapida.
- Utilizati modul `avansat` pentru audit, planificare si jurnal.
- Mentineti setul de comparatie limitat la molecule relevante.
- Marcati explicit moleculele aprobate inainte de export.
- Verificati mai intai verdictul rapid si comparatorul de piata, apoi intrati in detaliile RLVR.

## 11. Depanare

### Backend indisponibil

Daca interfata afiseaza erori la `/api/dashboard` sau `/api/control/start`, verificati pornirea backend-ului pe:

```text
http://127.0.0.1:8000
```

### Frontend incarcat, dar fara actualizari

- verificati daca sesiunea este pornita
- utilizati butonul `Actualizeaza`
- confirmati ca backend-ul raspunde

### Vizualizarea 3D este mai lenta

Vizualizarea 3D foloseste `3Dmol` si poate fi mai costisitoare decat vizualizarea 2D. Pentru analiza curenta si selectie rapida, modul `2D` este recomandat implicit in modul `simplificat`.

## 12. Concluzie

`OncoSynth` trebuie utilizat ca un instrument de decizie asistata pentru trierea si prioritizarea moleculelor, nu doar ca un dashboard de prezentare. Fluxul recomandat este: rezumat executiv, molecula activa, comparatie, risc, revizie, export.
