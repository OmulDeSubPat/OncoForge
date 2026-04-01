# Legenda explicativa pentru juriu non-tehnic

Acest folder contine un set scurt de grafice alese special pentru o prezentare clara, usor de urmarit si usor de explicat.
Ordinea fisierelor este gandita ca o poveste: problema, metoda, rezultate, validare si concluzie.

## 01_contextul_problemei.png
Ce arata:
Imaginea arata de ce tema este importanta din punct de vedere medical. Sunt prezentate valori mari care arata impactul cancerului pulmonar si al adenocarcinomului.

Cum o explici:
"Am pornit de la o problema reala si foarte mare. Numarul de cazuri este ridicat, iar asta justifica nevoia unei metode care sa ajute la selectia mai rapida a candidatilor promitori."

Mesaj cheie:
Proiectul raspunde unei nevoi reale, nu unei curiozitati teoretice.

## 02_cum_functioneaza_sistemul.png
Ce arata:
Acesta este fluxul proiectului in 5 pasi: colectam date, antrenam modelul, generam molecule, verificam candidatii si alegem lotul final.

Cum o explici:
"Nu incercam sa inlocuim laboratorul, ci sa reducem numarul de optiuni la un grup mult mai mic si mai bun de candidati."

Mesaj cheie:
Sistemul este un filtru inteligent care restrange spatiul de cautare.

## 03_rezultatele_pe_scurt.png
Ce arata:
Graficul rezuma proiectul in 4 numere usor de retinut: cate molecule au fost evaluate, cate au fost generate, cate au fost verificate si cate au ramas la final.

Cum o explici:
"Aici se vede scara proiectului si faptul ca procesul nu se opreste la generare. Din multe optiuni, doar un numar mic trece de filtre si ajunge in lotul final."

Mesaj cheie:
Rezultatul nu este doar cantitate, ci selectie riguroasa.

## 04_cum_restrangem_spatiul_chimic.png
Ce arata:
Acest grafic arata vizual cum reducem treptat numarul de molecule: de la mii de optiuni la un lot final foarte mic.

Cum o explici:
"Asta este una dintre ideile centrale ale proiectului. Nu putem testa totul, asa ca folosim AI pentru a merge de la foarte multe variante la cateva care merita atentie."

Mesaj cheie:
Proiectul economiseste timp si resurse prin prioritizare.

## 05_de_ce_abordarea_multi_agent.png
Ce arata:
Graficul compara o selectie simpla cu o selectie multi-agent. Se vede cate molecule bune sunt gasite si cate sunt ratate.

Cum o explici:
"Cand folosim mai multe filtre si perspective, gasim mai multe molecule bune decat daca ne uitam doar la un scor simplu."

Mesaj cheie:
Abordarea multi-agent imbunatateste selectia finala.

Atentie:
Nu spune "dovedeste superioritate absoluta". Mai corect este "arata un avantaj clar in acest test".

## 06_validare_pe_date_externe.png
Ce arata:
Graficul arata cum se comporta modelul pe surse externe de date, nu doar pe datele vazute in antrenare.

Cum o explici:
"Acesta este un test important de incredere. Nu ne uitam doar la ce stie modelul pe datele deja cunoscute, ci si la cat de bine generalizeaza."

Mesaj cheie:
Modelul nu este evaluat doar intern, ci si pe surse independente.

## 07_molecula_noastra_vs_standard.png
Ce arata:
Este o comparatie simpla intre un candidat al proiectului si un standard cunoscut, pe cateva criterii usor de inteles.

Cum o explici:
"Nu spunem ca am gasit un medicament mai bun, dar aratam ca unele molecule propuse de sistem pot fi competitive pe criterii importante."

Mesaj cheie:
Candidatii generati sunt suficient de buni incat sa merite comparati cu repere cunoscute.

Atentie:
Explica faptul ca aceasta comparatie este in silico, nu clinica.

## 08_eroarea_modelului.png
Ce arata:
Graficul arata eroarea modelului in mai multe scenarii de test. Regula simpla este chiar notata pe grafic: mai mic inseamna mai bine.

Cum o explici:
"Acesta este graficul prin care aratam cat de precis este modelul. Nu cer juriului sa retina formula, doar ideea ca eroarea este rezonabila si controlata."

Mesaj cheie:
Modelul are performanta suficient de buna pentru a fi folosit ca instrument de prioritizare.

## 09_de_unde_vin_moleculele_finale.png
Ce arata:
Graficul arata din ce surse interne ale pipeline-ului provin cele 18 molecule finale: selectie diversa, shortlist, optimizare, RL si generatie.

Cum o explici:
"Lotul final nu vine dintr-o singura idee. El combina mai multe strategii, ceea ce il face mai echilibrat si mai robust."

Mesaj cheie:
Pipeline-ul foloseste mai multe cai de explorare, nu o singura metoda rigida.

## 10_statusul_lotului_final.png
Ce arata:
Graficul arata cate molecule sunt gata pentru pasii urmatori si cate sunt inca in categoria de suport.

Cum o explici:
"Nu toate moleculele finale sunt la acelasi nivel de maturitate. Unele sunt mai pregatite pentru pasi urmatori, iar altele ofera sustinere si diversitate."

Mesaj cheie:
Rezultatul final este organizat si prioritizat, nu doar o lista fara ordine.

## Termeni simpli pe care ii poti folosi
AI:
un sistem care invata din date si ajuta la selectie.

Molecula candidata:
o varianta propusa ca posibil punct de plecare pentru teste ulterioare.

Validare pe date externe:
test pe date independente, pentru a vedea daca modelul ramane util si in afara setului pe care a invatat.

Multi-agent:
mai multe filtre sau perspective de evaluare care lucreaza impreuna.

In silico:
evaluare facuta cu ajutorul calculatorului, nu direct in laborator.

## Formulare recomandate
"Rezultatele sugereaza..."
"Modelul ajuta la prioritizarea candidatilor..."
"Aceasta selectie este una in silico..."
"Candidatii finali merita evaluare ulterioara..."

## Formulare de evitat
"Am descoperit medicamentul."
"Modelul vindeca..."
"Rezultatele garanteaza succesul in laborator."
