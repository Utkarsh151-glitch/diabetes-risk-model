# Data: Pima Indians Diabetes Database

`pima_diabetes.csv`: 768 rows, 8 features + `Outcome`, 23,290 bytes.

SHA-256: `c980e5e01dc1c554a358102b773f34b29599d0051ec5cf5b56ad1c6fcb3a23b0`

## Source

- **OpenML dataset 37** ("diabetes", version 1): <https://www.openml.org/d/37>. Fetched on 9 Oct 2026 with `sklearn.datasets.fetch_openml(data_id=37, as_frame=True)`, columns renamed (table below), class `tested_positive` → `Outcome = 1`.
- **Original owners:** US National Institute of Diabetes and Digestive and Kidney Diseases. **Donor:** Vincent Sigillito, Applied Physics Laboratory, Johns Hopkins University (received 9 May 1990). Both from the OpenML description.
- OpenML's "original data" link points to the UCI Machine Learning Repository page (<https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes>), which returns 404 as of 9 Oct 2026.
- **Cross-check:** every value is identical, row by row, to two other public copies:
  - <https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv>
  - <https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv>

## License

OpenML lists the licence for dataset 37 as **"Public"** (the `licence` field of <https://www.openml.org/api/v1/json/data/37>). Its description asks users to follow the UCI citation policy.

## Population

From the source description: "all patients here are females at least 21 years old of Pima Indian heritage", and "the population lives near Phoenix, Arizona, USA". `Outcome = 1` means the patient showed signs of diabetes by World Health Organization criteria (2-hour post-load plasma glucose of at least 200 mg/dl at any survey examination, or found during routine medical care).

Class balance: 500 negative, 268 positive (34.9%). No exact duplicate rows.

## Columns

| Column | OpenML name | Meaning (from the source) | Unit |
|---|---|---|---|
| `Pregnancies` | `preg` | Number of times pregnant | count |
| `Glucose` | `plas` | Plasma glucose concentration 2 hours into an oral glucose tolerance test. **Not a fasting reading.** | mg/dL (the source gives no unit; its WHO cut-off of 200 mg/dl is on this scale) |
| `BloodPressure` | `pres` | Diastolic blood pressure | mmHg |
| `SkinThickness` | `skin` | Triceps skin fold thickness | mm |
| `Insulin` | `insu` | 2-hour serum insulin | µU/mL |
| `BMI` | `mass` | Body mass index: weight in kg / (height in m)² | kg/m² |
| `DiabetesPedigreeFunction` | `pedi` | Diabetes pedigree function (a family-history score defined in the original study) | none |
| `Age` | `age` | Age | years |
| `Outcome` | `class` | 1 = tested positive for diabetes, 0 = negative | |

## Zeros that mean "missing"

The source says "Missing Attribute Values: None", but a value of 0 is physiologically impossible in five columns, so those zeros are treated as missing (`src/pipeline.py`, `ZERO_MEANS_MISSING`) and imputed with the training-split median inside the model pipeline:

| Column | Rows with 0 |
|---|---|
| `Glucose` | 5 |
| `BloodPressure` | 35 |
| `SkinThickness` | 227 |
| `Insulin` | 374 |
| `BMI` | 11 |

376 of the 768 rows (49%) have at least one of these. A 0 in `Pregnancies` (111 rows) is a real value and is kept.

## Citation

Smith J.W., Everhart J.E., Dickson W.C., Knowler W.C., Johannes R.S. (1988). "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus." *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261–265. PMCID: [PMC2245318](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/).
