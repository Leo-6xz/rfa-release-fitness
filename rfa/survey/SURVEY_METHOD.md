# Split-protocol survey: method

Sample frame: papers returned by six fixed web-search queries (listed below), run in November 2026, that
(a) were openly readable and (b) train and evaluate a classifier on the MSU-ORNL power-system attack datasets
(Borges Hink et al. 2014; Pan, Morris & Adhikari 2015). Theses and papers using other datasets were excluded.
Each paper's methods section was read and its main evaluation protocol coded as one of:
RANDOM (row-level random/stratified split or k-fold CV over rows), FILE (whole data files held out),
SCENARIO (whole scenarios held out), TEMPORAL (time-ordered or prequential), UNCLEAR (not stated).

Queries:
1. "power system attack dataset" Mississippi State machine learning detection accuracy
2. Hink Morris "power system disturbance and cyber-attack discrimination" dataset classifier
3. "ORNL power system" "cyber attack" dataset PMU machine learning
4. arxiv "power system" cyber attack detection "Mississippi State" dataset deep learning
5. MDPI energies OR sensors OR electronics "power system attack" dataset intrusion detection PMU 37 scenarios
6. smart grid "cyber-attack" "natural events" "no events" dataset classification machine learning PMU

Result (n = 15): RANDOM 10, UNCLEAR 3, TEMPORAL 1, FILE 1, SCENARIO 0.
Of the 10 RANDOM papers, 8 state random, stratified, seeded or k-fold assignment; 2 (Farrukh 2021, Husnoo FeDiSa 2023)
give only a ratio split of pooled rows without saying whether rows were shuffled (column random_assignment_stated).
The FILE paper (Wang 2020) tests on one whole file (5,069 rows = multiclass data2.csv) but does not say that file was
excluded from training; its main 97.9% result has no stated protocol.

Verification (2026-11-09): every quote, page and headline metric was checked against the published or preprint PDF
(column pdf_page). Keshk 2017 has a font-encoded text layer and was checked by OCR; Intriago 2023 table values were
read by OCR. Published venues and DOIs replace arXiv-only entries where a published version exists (column venue_doi).
Other evaluation problems noted in passing (not part of the coding): oversampling before the split (Murugesan 2024,
and probably Farrukh 2021) and overlapping sliding windows split at random (Kamoona 2026).

Limitations: convenience sample of open-access work, coded by one reader.
