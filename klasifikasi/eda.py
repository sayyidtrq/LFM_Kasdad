# %% [markdown]
# <p align="center">
#   <img src="https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg" width="200" height="200">
# </p>

# %% [markdown]
# **Anggota Kelompok Dadsak**
# 1. Sayyid Thariq Gilang Muttaqien (2306275714)
# 2. Rania Berliana
# 3. Rama Aditya Harmono
# 4. Bon bon

# %% [markdown]
# # 🏆 Proyek Akhir KASDAD 25/26  
# ## Topik: La Liga Football Matches (LFM)
# ---
# 
# ### Pendahuluan
# ---
# 
# Sepak bola bukan hanya permainan 90 menit, tetapi juga rangkaian peristiwa yang kompleks — mulai dari operan, tembakan, hingga strategi pertahanan yang menentukan hasil akhir pertandingan.  
# Dalam konteks **La Liga**, salah satu liga paling kompetitif di dunia, setiap aksi di lapangan menciptakan jejak data yang merekam dinamika taktik dan performa tim.
# 
# Dataset **La Liga Football Matches (LFM)** menyediakan dua sumber utama:
# 1. **Matches Dataset** — berisi informasi umum tiap pertandingan, termasuk hasil, skor, penguasaan bola (*ball possession*), serta detail manajer, wasit, dan stadion.
# 2. **Events Dataset** — berisi data granular dari setiap aksi dalam pertandingan (seperti *pass, shot, dribble, tackle, interception,* dan *foul*).
# 
# Melalui kedua dataset ini, proyek ini bertujuan untuk:
# - Menggali **insight taktis dan performa tim** berdasarkan data pertandingan dan event.  
# - Membangun **model klasifikasi** untuk memprediksi hasil pertandingan (`Result`: Win/Draw/Lose).  
# - Membangun **model regresi** untuk memprediksi tingkat penguasaan bola (`Poss`).  
# - Melakukan **clustering** untuk mengelompokkan pertandingan berdasarkan gaya bermain atau karakteristik statistik.
# 
# ### Tujuan Analisis
# ---
# 
# 1. Siapakah pemain Barcelona yang kerap menjadi pemecah kebuntuan tim (ketika hasil masih defisit atau seri bagi tim yang bersangkutan) dalam dataset ini?
# 2. Apakah suatu formasi tertentu cenderung meng-counter formasi tim tertentu? Hubungkan dengan berbagai hal, misalnya pergerakan ofensif dan pengerakan defensif tim.
# 3. Tentukan gaya permainan tim mayoritas yang digunakan oleh Barcelona dari tahun ke tahun dari data yang ada. (Beberapa playstyle yang terkenal seperti: Possession Game, Quick Counter, Long Ball Counter, Out Wide, Long Ball; di luar ini, juga banyak, bisa dieksplorasi lebih lanjut)
# 4. Bagaimanakah pola defensif tim Barcelona dari tahun ke tahun, apakah ada penyesuaian pola jika menghadapi tim-tim tertentu?
# 
# 
# ### Relevansi
# ---
# 
# Proyek ini tidak hanya menggabungkan analisis statistik dan machine learning, tetapi juga memanfaatkan **domain knowledge sepak bola** untuk memahami data dalam konteks taktik dan strategi.  
# Dengan demikian, hasil analisis diharapkan dapat menjawab pertanyaan mendasar dalam analisis performa olahraga, seperti:
# > *Apakah dominasi penguasaan bola selalu berujung pada kemenangan?*  
# > *Apa ciri khas permainan Barcelona dari tahun ke tahun menurut data?*
# 

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as scp
from collections import Counter

pd.set_option('display.max_rows', None)     # tampilkan semua baris
pd.set_option('display.max_columns', None)  # tampilkan semua kolom
pd.set_option('display.width', None)        # biar nggak wrap otomatis
pd.set_option('display.max_colwidth', None) # biar kolom teks panjang tampil penuh

# %%

# Load dataset
path = 'train.csv'
test_path = 'test.csv'
events_path = 'events_train.csv'
startingxi = 'startingxi.csv'
positions = 'position_guide.csv'

match_data = pd.read_csv(path)
events_data = pd.read_csv(events_path)
starting_xi = pd.read_csv(startingxi)
positions_guide = pd.read_csv(positions)
test_data = pd.read_csv(test_path)

# %%
events_data['match_id'] = events_data['match_id'].astype(int)
match_data['match_id'] = match_data['match_id'].astype(int)

# pilih hanya kolom match_id + home_team.home_team_name lalu rename untuk kemudahan
match_home = match_data[['match_id', 'home_team.home_team_name','away_team.away_team_name']].rename(
    columns={'home_team.home_team_name': 'home_team_name',
             'away_team.away_team_name': 'away_team_name'}

)

# merge ke events (left join agar semua events tetap ada)
events_merged = events_data.merge(match_home, on='match_id', how='left')

# quick check
print(events_merged[['match_id','home_team_name','away_team_name']].tail(50))

# %% [markdown]
# > ## 1.) Exploratory Data Analysis (EDA) 

# %% [markdown]
# > helper functions

# %%
def missing_values_info(df,arr):
    for col in df:
        missing = df[col].isna().sum()
        arr.append([col, missing, (missing/len(events_data))*100])

missing_info = []
#sortir informasi missing values
missing_values_info(events_merged, missing_info)
missing_df = pd.DataFrame(missing_info, columns=['Column', 'Missing Values', 'Percentage'])
missing_df = missing_df.sort_values(by='Missing Values', ascending=False)
print(missing_df)

# %%
def drop_cols_by_names (df, col_names):
    return df.drop(columns=col_names)

# %% [markdown]
# > membuat klasifikasi untuk kolom - kolom pada dataset events berdasarkan jenis event sepeti `pass`, `shot`, `dribble`, serta `saved` yg diambil dari `goalkeeper`.

# %%
goalkeeper_cols = ['goalkeeper.success_out','goalkeeper.lost_out','shot.redirect','goalkeeper.lost_in_play','goalkeeper.success_in_play','goalkeeper.shot_saved_off_target','goalkeeper.shot_saved_to_post',
                   'goalkeeper.punched_out','goalkeeper.body_part.name','goalkeeper.technique.name','goalkeeper.outcome.name','goalkeeper.end_location','goalkeeper.position.name','goalkeeper.type.name']
passing_cols = ['pass.miscommunication','pass.deflected','pass.no_touch','pass.cut_back','pass.goal_assist','pass.outswinging','pass.inswinging','pass.through_ball','pass.aerial_won','pass.technique.name',
                'pass.shot_assist','pass.cross','pass.switch','pass.type.name','pass.outcome.name','pass.recipient.name','pass.body_part.name','pass.height.name','pass.angle','pass.end_location']
shoting_cols = ['shot.follows_dribble','shot.saved_off_target','shot.saved_to_post','shot.open_goal','shot.deflected','shot.aerial_won','shot.aerial_won','shot.first_time','shot.technique.name','shot.type.name',
                'shot.end_location','shot.body_part.name','shot.outcome.name']
dribble_cols = ['dribble.outcome.name','dribble.nutmeg','dribble.overrun','dribble.no_touch']


# %% [markdown]
# > melakukan pembuangan kolom yang memiliki persentase missing values yang besar serta tidak relevan untuk dilanjukan ke tahap preprocessing dan modeling.

# %%
high_col_percent_missing =['goalkeeper.success_out','goalkeeper.lost_out','goalkeeper.lost_in_play','shot.saved_off_target','shot.follows_dribble','shot.saved_off_target','goalkeeper.shot_saved_off_target','goalkeeper.punched_out','dribble.no_touch','pass.cut_back','pass.outswinging',
                           'substitution.replacement.name','substitution.outcome.name',
                           'goalkeeper.body_part.name','goalkeeper.technique.name','out','shot.body_part.name',
                           'pass.recipient.name','pass.body_part.name','pass.height.name','off_camera','shot.first_time','shot.redirect','shot.follows_dribble',
                           'dribble.nutmeg','dribble.overrun','dribble.no_touch','clearance.other','injury_stoppage.in_chain','goalkeeper.position.name',
                           'goalkeeper.type.name','timestamp','type.name','duel.type.name','pass.type.name','id','player.name','related_events',
                           'play_pattern.name','location','pass.angle','pass.end_location','shot.end_location','shot.technique.name','pass.technique.name',
                           'shot.type.name','shot.deflected','pass.shot_assist','pass.cross','pass.switch','pass.through_ball','pass.aerial_won','pass.miscommunication','pass.deflected','pass.no_touch']

copied_df = drop_cols_by_names(events_merged, high_col_percent_missing)

missing_values = []
missing_values_info(copied_df,missing_values)

index_output = list(range(0, len(missing_values)))
(pd.DataFrame(missing_values, columns=['Fitur', '# Missing','% Missing'], index=index_output).sort_values(by=['% Missing'], ascending=False, ignore_index=True))
 


# %% [markdown]
# > Melakukan konversi tipe data pada beberapa kolom yang sesuai dengan tipe data yang diinginkan.
# 

# %%
copied_df.dtypes

# %%
bool_cols = [
    'under_pressure', 'counterpress', 'pass.switch', 'pass.cross', 'pass.goal_assist',
    'pass.shot_assist', 'shot.first_time', 'shot.open_goal', 'shot.one_on_one',
    'clearance.left_foot', 'clearance.right_foot', 'clearance.head',
    'foul_committed.penalty', 'foul_won.penalty'
]

for col in bool_cols:
    events_data[col] = events_data[col].apply(lambda x: 1 if x == True or x == 'TRUE' else 0)

copied_df.dtypes

# %% [markdown]
# > Melakukan `feature engineering` berdasarkan domain knowledge sepak bola untuk menambah kolom - kolom baru yang dapat membantu dalam analisis dan pemodelan selanjutnya.

# %%
from sklearn.preprocessing import LabelEncoder

cat_cols = ['shot.outcome.name', 'pass.outcome.name', 'duel.outcome.name','dribble.outcome.name',
            'interception.outcome.name','block.save_block','goalkeeper.shot_saved_to_post','goalkeeper.outcome.name',
            'foul_won.penalty','foul_committed.penalty']
le = LabelEncoder()
for col in cat_cols:
    copied_df[col] = copied_df[col].astype(str).fillna('Unknown')
    copied_df[col] = le.fit_transform(copied_df[col])

copied_df.dtypes


# %%
copied_df.shape

# %%
copied_df.tail(100)

# %% [markdown]
# ### Melakukan understanding data terhadap dataset events

# %% [markdown]
# #### helper functions

# %%
for col in cat_cols:
    print(f"Value counts for {col}:")
    print(events_data[col].value_counts())
    print()

# %%
for col in cat_cols:
    print(f"Value counts for {col}:")
    print(copied_df[col].value_counts())
    print()

15,12,9,1,0,11,14

# %%
def is_barca(name: str) -> bool:
    return isinstance(name,str) and ('barcelona' in name.lower())

matches_barca = match_data[
    match_data['home_team.home_team_name'].apply(is_barca) |
    match_data['away_team.away_team_name'].apply(is_barca)
].copy()

# %% [markdown]
# #### understanding data startingxi.csv dan positions_guide.csv

# %%
# starting xi for team.name 'Barcelona'
starting_xi_barca = starting_xi[starting_xi['team.name'].apply(is_barca)].copy()
starting_xi_barca = starting_xi_barca.drop_duplicates(subset=['match_id', 'team.name']).reset_index(drop=True)
starting_xi_barca.head(5)

pos_cols = [c for c in starting_xi.columns if c.startswith('pos')]

rows = []
for _, r in starting_xi_barca.iterrows():
    for i in range(1, 26):
        pcol = f'pos{i}'
        player = r.get(pcol, None)
        if pd.notna(player) and str(player).strip() != '':
            rows.append({
                'match_id': r['match_id'],
                'team': r['team.name'],
                'position': i,
                'player': player,
            })

barca_starting_long = pd.DataFrame(rows)


# %%
positions_guide_copy = drop_cols_by_names(positions_guide,col_names='Position Name')
positions_guide_copy.rename(columns={'Position Abbreviation':'Position Name'}, inplace=True)
positions_guide_copy = positions_guide_copy.rename(columns={'Position Number': 'position', 'Position Name': 'position_name'})

positions_guide_copy.head(25)

barca_starting_long = barca_starting_long.merge(
    positions_guide_copy,
    on='position',
    how='left'
)

barca_starting_long.head(100)

# %% [markdown]
# > aggregasi data events berdasarkan match id dan team id untuk mendapatkan kolom - kolom statistik tim pada setiap pertandingan yang nantinya akan digunakan untuk preprocessing dan modeling.

# %%
copied_df['dribble.successful'] = copied_df['dribble.outcome.name'].apply(lambda x: 1 if x == 1 else 0)
copied_df['shot.outcome.goal'] = copied_df['shot.outcome.name'].apply(lambda x: 1 if x == 1 else 0)
copied_df['interception.outcome.positive'] = copied_df['interception.outcome.name'].apply(lambda x: 1 if x == 2 or x == 4 else 0)
copied_df['goalkeeper.outcome.positive'] = copied_df['goalkeeper.outcome.name'].apply(lambda x: 1 if x in [15,12,9,1,0,11,14] else 0)
copied_df['total_clearance'] = copied_df[['clearance.left_foot', 'clearance.right_foot', 'clearance.head']].sum(axis=1)
copied_df['foul_won.penalty'] = copied_df['foul_won.penalty'].apply(lambda x: 1 if x == 1 else 0)
copied_df['foul_committed.penalty'] = copied_df['foul_committed.penalty'].apply(lambda x: 1 if x == 1 else 0)

# %%
team_candidates = ['possession_team.name']
team_col = next((c for c in team_candidates if c in copied_df.columns), None)

df = copied_df.copy()
df['_team_norm'] = df[team_col].astype(str).str.strip().str.lower()
df['_home_norm'] = df['home_team_name'].astype(str).str.strip().str.lower()
df['_away_norm'] = df['away_team_name'].astype(str).str.strip().str.lower()

home_events = df[df['_team_norm'] == df['_home_norm'].copy()]
away_events = df[df['_team_norm'] == df['_away_norm'].copy()]
print(f"Home events: {len(home_events)}, Away events: {len(away_events)}")


# %%
copied_df['event_positive'] = (
    copied_df['pass.goal_assist'] + 
    copied_df['shot.outcome.goal'] + 
    copied_df['dribble.successful'] + 
    copied_df['interception.outcome.positive'] + 
    copied_df['goalkeeper.outcome.positive'] + 
    copied_df['foul_won.penalty']
)

copied_df['event_negative'] = (
    copied_df['foul_committed.penalty'] +
    (1 - copied_df['dribble.successful']) + 
    (copied_df['pass.outcome.name'] != 0).astype(int)  # jika 0 = sukses
)


# %%
agg_cols = [
    'event_positive', 'event_negative',
    'dribble.successful', 'pass.goal_assist', 'shot.outcome.goal',
    'goalkeeper.outcome.positive', 'interception.outcome.positive',
    'foul_won.penalty', 'foul_committed.penalty', 'total_clearance'
]

agg_home = copied_df.groupby(['match_id', 'home_team_name'])[agg_cols].sum().reset_index()
agg_away = copied_df.groupby(['match_id', 'away_team_name'])[agg_cols].sum().reset_index()

agg_home = agg_home.add_prefix('home_')
agg_away = agg_away.add_prefix('away_')

# gabungkan berdasarkan match_id
agg_match = pd.merge(
    agg_home, agg_away, left_on='home_match_id', right_on='away_match_id', how='outer', suffixes=('', '_y')
)
agg_match = agg_match.rename(columns={'home_match_id': 'match_id'})


# %%
agg_match['pos_event_diff'] = agg_match['home_event_positive'] - agg_match['away_event_positive']
agg_match['neg_event_diff'] = agg_match['home_event_negative'] - agg_match['away_event_negative']

agg_match['attack_efficiency_home'] = agg_match['home_shot.outcome.goal'] / (agg_match['home_event_positive'] + 1e-5)
agg_match['attack_efficiency_away'] = agg_match['away_shot.outcome.goal'] / (agg_match['away_event_positive'] + 1e-5)

agg_match['defense_efficiency_home'] = agg_match['home_goalkeeper.outcome.positive'] / (agg_match['home_event_negative'] + 1e-5)
agg_match['defense_efficiency_away'] = agg_match['away_goalkeeper.outcome.positive'] / (agg_match['away_event_negative'] + 1e-5)
agg_match.head(10)

# %%

# pastikan kolom identifikasi tim & match
ev = copied_df.copy()

team_col = next((c for c in ['possession_team.name', 'team.name', 'team', 'possession_team'] if c in ev.columns), None)
if team_col is None:
    raise ValueError("Kolom team tidak ditemukan di events_data.")

ev['_team_norm'] = ev[team_col].astype(str).str.strip().str.lower()
ev['_home_norm'] = ev['home_team_name'].astype(str).str.strip().str.lower()
ev['_away_norm'] = ev['away_team_name'].astype(str).str.strip().str.lower()

ev['is_home'] = ev['_team_norm'] == ev['_home_norm']
ev['is_away'] = ev['_team_norm'] == ev['_away_norm']

ev['is_goal'] = ev['shot.outcome.name'].astype(str).str.contains('goal', case=False, na=False).astype(int)

agg_team = ev.groupby(['match_id', team_col]).agg(
    duration_mean=('duration', 'mean'),
    under_pressure_sum=('under_pressure', 'sum'),
    counterpress_sum=('counterpress', 'sum'),
    successful_passes=('pass.outcome.name', lambda x: (x == 0).sum()),
    successful_dribbles=('dribble.outcome.name', lambda x: (x == 0).sum()),
    interceptions=('interception.outcome.positive', 'sum'),
    goalkeeper_positive=('goalkeeper.outcome.positive', 'sum'),
    total_clearance=('total_clearance', 'sum'),
    foul_committed=('foul_committed.penalty', 'sum'),
    foul_won=('foul_won.penalty', 'sum'),
    goals=('is_goal', 'sum')
).reset_index()

agg_team = agg_team.merge(
    ev[['match_id', team_col, 'is_home', 'is_away']].drop_duplicates(),
    on=['match_id', team_col],
    how='left'
)

# ✅ merge home vs away pakai match_id
agg_home = agg_team[agg_team['is_home']]
agg_away = agg_team[agg_team['is_away']]

aggregated_features = pd.merge(
    agg_home,
    agg_away,
    on='match_id',
    how='outer',
    suffixes=('_home', '_away')
)

# ⚠️ JANGAN FLATTEN DULU — biarkan match_id tetap ada
# aggregated_features.columns = ['_'.join(col).strip('_') for col in aggregated_features.columns.values]  <-- hapus / komentarin

# lanjut bagian barca / non-barca
ev = events_data.copy()

# tanda goal: cek kolom teks dulu
if 'shot.outcome.name' in ev.columns:
    ev['shot_goal_flag'] = ev['shot.outcome.name'].astype(str).str.contains('goal', case=False, na=False).astype(int)
elif 'shot.outcome.goal' in ev.columns:
    ev['shot_goal_flag'] = ev['shot.outcome.goal'].fillna(0).astype(int)
else:
    ev['shot_goal_flag'] = 0

team_candidates = ['possession_team.name', 'team.name', 'team', 'possession_team']
team_col = next((c for c in team_candidates if c in ev.columns), None)

if team_col is None:
    print("Warning: team column not found in events_data — cannot compute Barcelona goals per match.")
    aggregated_features['shot.outcome.goal.barca'] = 0
    aggregated_features['shot.outcome.goal.nonbarca'] = 0
else:
    ev['is_barca'] = ev[team_col].astype(str).str.contains('barcelona', case=False, na=False)
    ev['is_nonbarca'] = ~ev['is_barca']

    barca_goals_per_match = (
        ev.loc[ev['is_barca'] & (ev['shot_goal_flag'] == 1)]
        .groupby('match_id')['shot_goal_flag']
        .sum()
        .rename('shot.outcome.goal.barca')
        .reset_index()
    )

    nonbarca_goals_per_match = (
        ev.loc[ev['is_nonbarca'] & (ev['shot_goal_flag'] == 1)]
        .groupby('match_id')['shot_goal_flag']
        .sum()
        .rename('shot.outcome.goal.nonbarca')
        .reset_index()
    )

    aggregated_features = (
        aggregated_features
        .merge(barca_goals_per_match, on='match_id', how='left')
        .merge(nonbarca_goals_per_match, on='match_id', how='left')
        .fillna({'shot.outcome.goal.barca': 0, 'shot.outcome.goal.nonbarca': 0})
    )

aggregated_features.head(10)    

# %%
df_diff = aggregated_features.copy()

diff_features = [
    'duration_mean',
    'under_pressure_sum',
    'counterpress_sum',
    'successful_passes',
    'successful_dribbles',
    'interceptions',
    'goalkeeper_positive',
    'total_clearance',
    'foul_committed',
    'foul_won',
    'goals'
]

for f in diff_features:
    home_col = f"{f}_home"
    away_col = f"{f}_away"
    if home_col in df_diff.columns and away_col in df_diff.columns:
        df_diff[f"{f}_diff"] = df_diff[home_col].fillna(0) - df_diff[away_col].fillna(0)

df_diff["pass_efficiency_home"] = df_diff["successful_passes_home"] / (df_diff["successful_passes_home"] + df_diff["successful_passes_away"] + 1e-5)
df_diff["dribble_efficiency_home"] = df_diff["successful_dribbles_home"] / (df_diff["successful_dribbles_home"] + df_diff["successful_dribbles_away"] + 1e-5)
df_diff["defense_efficiency_home"] = df_diff["goalkeeper_positive_home"] / (df_diff["goalkeeper_positive_home"] + df_diff["goalkeeper_positive_away"] + 1e-5)

if "shot.outcome.goal.barca" in df_diff.columns and "shot.outcome.goal.nonbarca" in df_diff.columns:
    df_diff["barca_goal_diff"] = df_diff["shot.outcome.goal.barca"] - df_diff["shot.outcome.goal.nonbarca"]

print("✅ Difference & Efficiency features created!")
new_cols = [c for c in df_diff.columns if c.endswith("_diff") or c.endswith("_efficiency_home")]
print("New columns added:", new_cols[:15], "...")


aggregated_features = pd.merge(
    df_diff,
    aggregated_features[['match_id']],
    on='match_id',
    how='left'
)

aggregated_features.head(10)

# %%

df_merged = match_data.merge(
    aggregated_features, 
    on='match_id', 
    how='left'  # LEFT JOIN 
)

test_merged = test_data.merge(
    aggregated_features,
    on='match_id',
    how='left'  # LEFT JOIN
)

print("=" * 60)
print("HASIL MERGE MATCH_DATA + AGGREGATED_FEATURES")
print("=" * 60)
print(f"Match data shape: {match_data.shape}")
print(f"Aggregated features shape: {aggregated_features.shape}")
print(f"Merged data shape: {df_merged.shape}")

df_merged.head(100)

# %%
# Cek struktur data
print("Events data shape:", copied_df.shape)
print("\nMatch data shape:", match_data.shape)
print("\nEvents columns:", copied_df.columns.tolist()[:20])  # show first 20
print("\nMatch columns:", match_data.columns.tolist())

# %% [markdown]
# ## Pertanyaan - pertanyaan EDA
# ---

# %% [markdown]
# #### 1.  Bagaimanakah pola defensif tim Barcelona dari tahun ke tahun, apakah ada penyesuaian pola jika menghadapi tim-tim tertentu?

# %%
# Gunakan proxy defensive: duel counts, clearances, fouls
defensive_features = [c for c in df_merged.columns if any(k in c for k in ['duel','clearance','tackle','interception','conceded','nonbarca','goalkeeper'])]
print('Detected defensive-related columns:', defensive_features[:20])

if defensive_features:
    df_merged['match_date'] = pd.to_datetime(df_merged['match_date'])
    df_merged['year'] = df_merged['match_date'].dt.year
    def_summary = df_merged.groupby('year')[defensive_features].mean()
    print('\nAverage defensive metrics per year:')
    display(def_summary)

    # Plot trends for first few defensive metrics
    to_plot = defensive_features[:9]
    if to_plot:
        plt.figure(figsize=(12,9))
        for c in to_plot:
            if c in def_summary.columns:
                plt.plot(def_summary.index, def_summary[c], marker='o', label=c)
        plt.legend()
        plt.title('Defensive metrics trend per year')
        plt.xlabel('Year')
        plt.ylabel('Average per match')
        plt.grid(alpha=0.3)
        plt.show()
else:
    print('Tidak ditemukan kolom defensif yang jelas di df_merged. Pastikan aggregated defensive features dibuat saat agregasi events.')

# %%

# 1) Fokus hanya pertandingan Barcelona
barca_mask = (
    df_merged['home_team.home_team_name'].astype(str).str.contains('barcelona', case=False, na=False) |
    df_merged['away_team.away_team_name'].astype(str).str.contains('barcelona', case=False, na=False)
)
df_barca = df_merged.loc[barca_mask].copy()
if df_barca.empty:
    print("Tidak ada match Barcelona ditemukan di df_merged.")
else:
    # 2) Tentukan lawan dan apakah Barca main kandang
    df_barca['barca_is_home'] = df_barca['home_team.home_team_name'].astype(str).str.contains('barcelona', case=False, na=False)
    df_barca['opponent'] = np.where(df_barca['barca_is_home'], df_barca['away_team.away_team_name'], df_barca['home_team.home_team_name'])
    df_barca['opponent'] = df_barca['opponent'].fillna('Unknown')

    # 3) Defensive columns (gunakan fitur agregat defensif yang sudah dibuat)
    defensive_cols = [c for c in df_barca.columns if any(k in c for k in ['duel','clearance','tackle','interception','conceded','nonbarca','goalkeeper'])]
    if not defensive_cols:
        print("Tidak ditemukan kolom defensif di df_merged. Pastikan agregasi defensif dibuat sebelumnya.")
    else:
        print("Defensive columns detected:", defensive_cols)

        # 4) Ringkasan per opponent
        opp_summary = df_barca.groupby('opponent')[defensive_cols].agg(['mean','std','count'])
        # tampilkan top opponent berdasarkan jumlah match
        opp_counts = opp_summary[ (defensive_cols[0],'count') ].sort_values(ascending=False)
        top_opponents = opp_counts.head(30).index.tolist()
        print("\nTop opponents by matches (top 30):")
        display(opp_counts.head(30))

        print("\nDefensive summary (mean) for top opponents:")
        display(opp_summary.loc[top_opponents].xs('mean', level=1, axis=1).sort_values(by=defensive_cols[0], ascending=False))

        # 5) Visualisasi: bar chart defensive metric per opponent (top N)
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")

        top_n = top_opponents[:30]
        metrics_to_plot = defensive_cols[:4]  # pilih beberapa metric defensif utama
        for metric in metrics_to_plot:
            plt.figure(figsize=(10,5))
            vals = df_barca.groupby('opponent')[metric].mean().loc[top_n].fillna(0)
            vals.sort_values(ascending=False).plot(kind='bar')
            plt.title(f"Average {metric} vs Opponent (Barcelona matches) - top {len(top_n)} opponents")
            plt.ylabel(f"Average {metric}")
            plt.xlabel("Opponent")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        # 6) Trend per year untuk top opponents (lihat apakah ada adjustment)
        df_barca['match_date'] = pd.to_datetime(df_barca['match_date'], errors='coerce')
        df_barca['year'] = df_barca['match_date'].dt.year.fillna(0).astype(int)
        years = sorted(df_barca['year'].unique())

        for metric in metrics_to_plot:
            plt.figure(figsize=(10,6))
            for opp in top_n:
                series = df_barca[df_barca['opponent']==opp].groupby('year')[metric].mean().reindex(years)
                if series.notna().sum() > 0:
                    plt.plot(series.index, series.values, marker='o', label=opp)
            plt.title(f"Yearly trend of {metric} vs top opponents")
            plt.xlabel("Year")
            plt.ylabel(metric)
            plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()


# %% [markdown]
# #### 2. Tentukan gaya permainan tim mayoritas yang digunakan oleh Barcelona dari tahun ke tahun dari data yang ada. (Beberapa playstyle yang terkenal seperti: Possession Game, Quick Counter, Long Ball Counter, Out Wide, Long Ball; di luar ini, juga banyak, bisa dieksplorasi lebih lanjut)

# %%
playsyles = ['Possesion Game', 'Quick Counter', 'Long Ball Counter', 'Out Wide', 'Long Ball', 'High Press', 'Tiki Taka', 'Direct Play', 'Wing Play', 'Park the Bus']

cols_possession = [
    "match_id",
    "team.name",
    "player.name",
    "timestamp",
    "period",
    "possession",
    "duration",
    "type.name",
    "location",
    "pass.recipient.name",
    "pass.angle",
    "pass.height.name",
    "pass.type.name",
    "pass.outcome.name",
    "pass.cross",
    "pass.through_ball",
    "pass.switch",
    "pass.cut_back",
    "carry.end_location",
    "dribble.outcome.name",
    "under_pressure",
    "counterpress",
    "duel.outcome.name",
    "interception.outcome.name",
    "goalkeeper.type.name",
    "goalkeeper.outcome.name"
]

barca_events = events_data[events_data["team.name"].astype(str).str.contains("barcelona", case=False, na=False)][cols_possession].copy()

# hitung agregat per possession — pakai .astype(str).str.lower() untuk hindari mismatch casing / NaN
possession_stats = (
    barca_events.groupby("possession")
    .agg(
        team=("team.name", "first"),
        duration=("duration", "sum"),
        total_passes=("type.name", lambda x: x.astype(str).str.lower().eq("pass").sum()),
        complete_passes=("pass.outcome.name", lambda x: x.astype(str).str.lower().eq("complete").sum()),
        short_passes=("pass.height.name", lambda x: x.astype(str).str.lower().eq("ground pass").sum()),
        carries=("type.name", lambda x: x.astype(str).str.lower().eq("carry").sum()),
        dribbles=("type.name", lambda x: x.astype(str).str.lower().eq("dribble").sum()),
        duels_lost=("duel.outcome.name", lambda x: x.astype(str).str.lower().eq("lost").sum()),
        duels_won=("duel.outcome.name", lambda x: x.astype(str).str.lower().eq("won").sum())
    )
    .reset_index()
)

# hindari pembagian oleh 0
possession_stats["pass_accuracy"] = possession_stats["complete_passes"] / possession_stats["total_passes"].replace(0, np.nan)
possession_stats["pass_accuracy"] = possession_stats["pass_accuracy"].fillna(0.0)

possession_stats["short_pass_ratio"] = possession_stats["short_passes"] / possession_stats["total_passes"].replace(0, np.nan)
possession_stats["short_pass_ratio"] = possession_stats["short_pass_ratio"].fillna(0.0)

possession_stats["passes_per_possession"] = possession_stats["total_passes"] / possession_stats["duration"].replace(0, np.nan)
possession_stats["passes_per_possession"] = possession_stats["passes_per_possession"].fillna(0.0)

# ...existing code...
barca_summary = possession_stats.groupby("team").agg(
    avg_possession_duration=("duration", "mean"),
    avg_pass_accuracy=("pass_accuracy", "mean"),
    avg_short_pass_ratio=("short_pass_ratio", "mean"),
    avg_passes_per_possession=("passes_per_possession", "mean"),
    avg_duel_success_rate=("duels_won", lambda x: x.sum() / (x.sum() + possession_stats["duels_lost"].sum()) if (x.sum() + possession_stats["duels_lost"].sum())>0 else 0)
)

display(barca_summary.head())

# %%
barca_matches = match_data[match_data["home_team.home_team_name"] == "Barcelona"]  # jika Barcelona home
match_data['match_date'] = pd.to_datetime(match_data['match_date'], errors='coerce')
if 'season' not in match_data.columns:
    match_data['season'] = match_data['match_date'].dt.year.astype('Int64')  # tahun sebagai season

# jika ingin filter match Barca (opsional)
barca_matches = match_data[match_data["home_team.home_team_name"].astype(str).str.contains("Barcelona", case=False, na=False)]

# merge barca_events dengan kolom season yang sekarang pasti ada
barca_events = barca_events.merge(match_data[['match_id', 'season']], on='match_id', how='left')

# Agregasi per musim/tahun
eda = (
    barca_events.groupby("season")
    .agg(
        passes=("type.name", lambda x: (x == "Pass").sum()),
        pass_complete=("pass.outcome.name", lambda x: (x == "Complete").sum()),
        pass_long=("pass.height.name", lambda x: (x == "High Pass").sum()),
        pass_short=("pass.height.name", lambda x: (x == "Ground Pass").sum()),
        carries=("type.name", lambda x: (x == "Carry").sum()),
        dribbles=("type.name", lambda x: (x == "Dribble").sum()),
        duels_won=("duel.outcome.name", lambda x: (x == "Won").sum()),
        duels_lost=("duel.outcome.name", lambda x: (x == "Lost").sum()),
        under_pressure=("under_pressure", lambda x: (x == True).sum()),
        counterpress=("counterpress", lambda x: (x == True).sum()),
        avg_duration=("duration", "mean")
    )
    .reset_index()
)

# Hitung metrik turunan
eda["pass_accuracy"] = eda["pass_complete"] / eda["passes"]
eda["short_pass_ratio"] = eda["pass_short"] / eda["passes"]
eda["long_pass_ratio"] = eda["pass_long"] / eda["passes"]
eda["duel_success_rate"] = eda["duels_won"] / (eda["duels_won"] + eda["duels_lost"])

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.lineplot(data=eda, x="season", y="pass_accuracy", ax=axes[0,0]).set_title("Akurasi Umpan per Musim")
sns.lineplot(data=eda, x="season", y="short_pass_ratio", ax=axes[0,1]).set_title("Proporsi Umpan Pendek")
sns.lineplot(data=eda, x="season", y="avg_duration", ax=axes[1,0]).set_title("Rata-rata Durasi Possession")
sns.lineplot(data=eda, x="season", y="counterpress", ax=axes[1,1]).set_title("Intensitas Counterpress")

plt.tight_layout()
plt.show()

# %%
def classify_playstyle(row):
    if row["pass_accuracy"] > 0.87 and row["short_pass_ratio"] > 0.75:
        return "Tiki Taka"
    elif row["pass_accuracy"] > 0.85 and row["avg_duration"] > 10:
        return "Possession Game"
    elif row["long_pass_ratio"] > 0.25 and row["avg_duration"] < 8:
        return "Long Ball Counter"
    elif row["counterpress"] > eda["counterpress"].mean() * 1.2:
        return "High Press"
    elif row["short_pass_ratio"] < 0.5 and row["avg_duration"] < 6:
        return "Direct Play"
    else:
        return "Mixed Playstyle"

eda["playstyle"] = eda.apply(classify_playstyle, axis=1)
eda[["season", "playstyle"]]


# %%
plt.figure(figsize=(10,5))
sns.countplot(data=eda, x="season", hue="playstyle", palette="Set2")
plt.title("Gaya Bermain Dominan Barcelona per Musim")
plt.xticks(rotation=45)
plt.show()


# %% [markdown]
# #### 3. Apakah suatu formasi tertentu cenderung meng-counter formasi tim tertentu? Hubungkan dengan berbagai hal, misalnya pergerakan ofensif dan pengerakan defensif tim.

# %%
# === Analisis formasi (jika ada kolom formation / play_pattern) ===
# Detect possible formation/play columns
formation_cols = [c for c in df_merged.columns if 'formation' in c.lower() or 'formation' in c]
play_pattern_cols = [c for c in df_merged.columns if 'play_pattern' in c.lower() or 'playstyle' in c.lower() or 'play_pattern.name' in c]

print('Detected formation columns:', formation_cols)
print('Detected play pattern columns:', play_pattern_cols)

if formation_cols:
    # Example: if home formation and away formation exist in match_data
    home_formation_col = None
    away_formation_col = None
    for c in formation_cols:
        if 'home' in c.lower():
            home_formation_col = c
        if 'away' in c.lower():
            away_formation_col = c
    # fallback: take first two if they exist
    if not home_formation_col and len(formation_cols) >= 1:
        home_formation_col = formation_cols[0]
    if not away_formation_col and len(formation_cols) >= 2:
        away_formation_col = formation_cols[1]

    if home_formation_col and away_formation_col:
        print(f'Using {home_formation_col} (home) and {away_formation_col} (away) to analyze formation matchups')
        ct = pd.crosstab(df_merged[home_formation_col], df_merged[away_formation_col])
        display(ct)
        plt.figure(figsize=(10,6))
        sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
        plt.title('Formation matchups: home vs away')
        plt.xlabel('Away formation')
        plt.ylabel('Home formation')
        plt.tight_layout()
        plt.show()
    else:
        print('Kolom formation tidak lengkap untuk analisis matchup (butuh home & away formation).')
else:
    print('Tidak ditemukan kolom formation di df_merged. Jika Anda punya data formation, tambahkan atau cek nama kolomnya.')

# If play_pattern columns exist, show distribution
if play_pattern_cols:
    for c in play_pattern_cols:
        print('\nDistribution of', c)
        display(df_merged[c].value_counts().head(20))
else:
    print('Tidak ditemukan kolom play_pattern / playstyle eksplisit di df_merged. play_pattern biasanya di events_data (kolom play_pattern.name) jika tersedia.')

# %% [markdown]
# #### 4. Siapakah pemain Barcelona yang kerap menjadi pemecah kebuntuan tim (ketika hasil masih defisit atau seri bagi tim yang bersangkutan) dalam dataset ini?

# %%
# Flexible detection of goal events & team column
team_col = None
for candidate in ['possession_team.name', 'team.name', 'team', 'possession_team']:
    if candidate in events_data.columns:
        team_col = candidate
        break

goal_col = None
if 'shot.outcome.name' in events_data.columns:
    goal_col = 'shot.outcome.name'
elif 'type.name' in events_data.columns:
    goal_col = 'type.name'

if team_col is None or goal_col is None or 'player.name' not in events_data.columns:
    print('Kolom yang dibutuhkan tidak lengkap di events_data. Diperlukan kolom: player.name, team identifier (possession_team.name atau team.name), dan kolom yang menandai goal (shot.outcome.name atau type.name).')
else:
    ev = events_data.copy().reset_index()
    ev['is_goal'] = ev[goal_col].astype(str).str.contains('goal', case=False, na=False)
    goals = ev[ev['is_goal']].copy()
    print(f'Terdapat {len(goals)} goal-event yang terdeteksi (berdasarkan kolom {goal_col}).')

    #display player.name,shot.outcome.name,team_col,match_id
    display(goals[['player.name', goal_col, team_col, 'match_id']].head(100))
    
    # Build timeline per match
    from collections import defaultdict
    scorer_counts = defaultdict(int)
    match_summaries = []

    for mid, gdf in goals.groupby('match_id'):
        gdf = gdf.sort_values('index')  # asumsi urutan kronologis sesuai index
        # Determine home/away team names from match_data
        row = match_data[match_data['match_id'] == mid]
        if row.empty:
            continue
        home = row.iloc[0].get('home_team.home_team_name', None)
        away = row.iloc[0].get('away_team.away_team_name', None)

        # track scores
        score = {home:0, away:0}
        # iterate goal events
        for _, evrow in gdf.iterrows():
            team = evrow[team_col]
            player = evrow.get('player.name', None)
            if pd.isna(team) or pd.isna(player):
                continue
            # before the goal
            opp = home if team == away else away
            team_before = score.get(team, 0)
            opp_before = score.get(opp, 0)
            # condition: previously trailing or tied
            if team_before <= opp_before:
                # count if this team is Barcelona (either home or away)
                if (str(team).lower() == 'barcelona') or (home == 'Barcelona' and team==home) or (away == 'Barcelona' and team==away):
                    scorer_counts[player] += 1
            # update score (assume goal increments by 1)
            score[team] = team_before + 1

    # Show top players
    scorer_series = pd.Series(scorer_counts).sort_values(ascending=False)
    print('\nTop pemain yang menjadi pemecah kebuntuan (hanya Barcelona):')
    if scorer_series.empty:
        print('Tidak ditemukan goal-event yang memenuhi kondisi atau kolom team names tidak konsisten.')
    else:
        display(scorer_series.head(20))
        scorer_series.head(20).plot(kind='bar', figsize=(10,5))
        plt.title('Top Pemain Pemecah Kebuntuan untuk Barcelona (count)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

# %% [markdown]
# berdasarkan analisis EDA poin 4 di atas, dapat disimpulkan bahwa pemain Barcelona yang kerap menjadi pemecah kebuntuan tim dalam dataset ini adalah **Lionel Messi** dengan total 39 events (goal contribution events) yang berkontribusi pada perubahan skor ketika tim dalam kondisi defisit atau seri. Disusul oleh **Luis Suárez** dengan 19 events, dan **Arturo Vidal.** dengan 7 events.

# %% [markdown]
# #### 5. Apakah possession yang lebih tinggi berkorelasi dengan xG / shots on target / peluang bersih? 

# %%
# NEW CELL: Integrasi pipeline untuk menjawab: apakah possession berkorelasi dengan xG / shots on target / peluang bersih?
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy import stats

sns.set(style="whitegrid")

# use existing dataframes
ev = events_data.copy()
md = match_data.copy()

# detect team col and possession col in events (fallbacks)
team_col = next((c for c in ['team.name','possession_team.name','team'] if c in ev.columns), None)
pos_col = next((c for c in ['possession','possession_pct','possession_pct_home','possession_home'] if c in ev.columns), None)

# detect shot type/outcome cols and candidate xg/big chance cols
shot_type_col = next((c for c in ['type.name','event.type'] if c in ev.columns), None)
shot_outcome_col = next((c for c in ['shot.outcome.name','shot.outcome'] if c in ev.columns), None)
xg_col = next((c for c in ev.columns if 'xg' in c.lower()), None)
big_chance_col = next((c for c in ev.columns if 'big' in c.lower() or 'big_chance' in c.lower() or 'clear_chance' in c.lower()), None)

if team_col is None:
    raise RuntimeError("Team column not found in events_data; needed for per-team aggregation.")

# --- 1) Build xG on shot-events if no xg column available ---
def extract_xy(val):
    try:
        if isinstance(val, (list, tuple)):
            return float(val[0]), float(val[1])
        s = str(val).strip().strip('[]() ')
        parts = [p.strip() for p in s.split(',') if p.strip()!='']
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except:
        pass
    return (np.nan, np.nan)

# ensure loc_x/loc_y present
if 'loc_x' not in ev.columns or 'loc_y' not in ev.columns:
    if 'shot.end_location' in ev.columns:
        ev[['loc_x','loc_y']] = ev['shot.end_location'].apply(lambda v: pd.Series(extract_xy(v)))
    elif 'location' in ev.columns:
        ev[['loc_x','loc_y']] = ev['location'].apply(lambda v: pd.Series(extract_xy(v)))
    elif 'location_x' in ev.columns and 'location_y' in ev.columns:
        ev['loc_x'] = pd.to_numeric(ev['location_x'], errors='coerce')
        ev['loc_y'] = pd.to_numeric(ev['location_y'], errors='coerce')

# detect shot rows robustly
def is_shot_row(r):
    if shot_type_col and isinstance(r.get(shot_type_col), str) and 'shot' in r.get(shot_type_col).lower():
        return True
    if shot_outcome_col and isinstance(r.get(shot_outcome_col), str):
        if any(k in r.get(shot_outcome_col).lower() for k in ('shot','goal','on target','saved')):
            return True
    return False

ev['_is_shot'] = ev.apply(is_shot_row, axis=1)
shots = ev[ev['_is_shot']].copy()
shots = shots.dropna(subset=['loc_x','loc_y'], how='any').copy()

if not shots.empty and xg_col is None:
    # compute simple distance & angle features
    x_min, x_max = shots['loc_x'].min(), shots['loc_x'].max()
    y_min, y_max = shots['loc_y'].min(), shots['loc_y'].max()
    y_center = (y_min + y_max) / 2.0
    # estimate goal half-width in data units
    goal_half = (y_max - y_min) * (7.32 / 68.0) / 2.0
    left_goal_x, right_goal_x = x_min, x_max

    def shot_feats(r):
        x, y = r['loc_x'], r['loc_y']
        gx = left_goal_x if abs(x-left_goal_x) < abs(x-right_goal_x) else right_goal_x
        gyc = y_center
        dx = gx - x
        dy = gyc - y
        distance = math.hypot(dx, dy)
        top_post = (gx, gyc + goal_half)
        bot_post = (gx, gyc - goal_half)
        a1 = math.atan2(top_post[1] - y, top_post[0] - x)
        a2 = math.atan2(bot_post[1] - y, bot_post[0] - x)
        angle = abs(a1 - a2)
        is_header = 0
        for b in ['shot.body_part','shot.body_part.name','body_part.name','body_part']:
            if b in r and pd.notna(r[b]) and isinstance(r[b], str) and 'head' in r[b].lower():
                is_header = 1
                break
        under_pressure = int(r['under_pressure']) if 'under_pressure' in r and pd.notna(r['under_pressure']) else 0
        counterpress = int(r['counterpress']) if 'counterpress' in r and pd.notna(r['counterpress']) else 0
        return pd.Series([distance, angle, is_header, under_pressure, counterpress],
                         index=['distance','angle','is_header','under_pressure','counterpress'])
    feat_df = shots.apply(shot_feats, axis=1)
    shots = pd.concat([shots.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    # label is_goal
    if 'shot.outcome.goal' in shots.columns:
        shots['is_goal'] = shots['shot.outcome.goal'].fillna(0).astype(int)
    elif shot_outcome_col:
        shots['is_goal'] = shots[shot_outcome_col].astype(str).str.contains('goal', case=False, na=False).astype(int)
    else:
        shots['is_goal'] = 0

    features = ['distance','angle','is_header','under_pressure','counterpress']
    X = shots[features].fillna(0).values
    y = shots['is_goal'].values

    if y.sum() >= 5 and len(y) >= 30:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        try:
            scores = cross_val_score(clf, Xs, y, cv=5, scoring='roc_auc')
            print("xG model CV AUC (5-fold):", np.round(scores,3))
        except Exception:
            pass
        clf.fit(Xs, y)
        shots['xG'] = clf.predict_proba(Xs)[:,1]
        # merge back xG into ev (align by index if possible)
        ev = ev.merge(shots[['loc_x','loc_y','xG']], how='left', left_on=['loc_x','loc_y'], right_on=['loc_x','loc_y'])
        # if merge created duplicates, prefer shots.xG
        if 'xG' not in ev.columns:
            ev['xG'] = np.nan
    else:
        print("Tidak cukup data untuk melatih xG model => xG akan bernilai NaN.")
        shots['xG'] = np.nan
        ev['xG'] = np.nan
else:
    # xg already present or no shots
    if xg_col:
        ev['xG'] = ev[xg_col]
    else:
        ev['xG'] = np.nan

# --- 2) Aggregate per match_id + team ---
# create shot on target proxy
def is_on_target(v):
    if pd.isna(v): return 0
    s = str(v).lower()
    return int(('on target' in s) or ('goal' in s) or ('saved' in s))
if shot_outcome_col:
    ev['_shot_on_target'] = ev[shot_outcome_col].apply(is_on_target)
else:
    ev['_shot_on_target'] = 0
# if big chance col exists, use as peluang_bersih proxy
if big_chance_col:
    ev['_big_chance'] = ev[big_chance_col].fillna(0).astype(int)
else:
    ev['_big_chance'] = ev['_shot_on_target']  # fallback

group_cols = ['match_id', team_col]
agg = ev.groupby(group_cols).agg(
    possession_mean = (pos_col, 'mean') if pos_col in ev.columns else ('duration','sum'),
    shots = ('_is_shot','sum'),
    shots_on_target = ('_shot_on_target','sum'),
    xG_sum = ('xG','sum'),
    big_chances = ('_big_chance','sum')
).reset_index()

# normalize possession if >1 (some datasets store 0-1)
if agg['possession_mean'].notna().any():
    if agg['possession_mean'].max() <= 1.01:
        agg['possession_pct'] = agg['possession_mean'] * 100
    else:
        agg['possession_pct'] = agg['possession_mean']

# clean NaNs
agg[['shots','shots_on_target','big_chances']] = agg[['shots','shots_on_target','big_chances']].fillna(0)
agg['xG_sum'] = agg['xG_sum'].fillna(0)

display(agg.head(10))

# --- 3) Correlations (possession_pct vs metrics) ---
metrics = ['shots','shots_on_target','xG_sum','big_chances']
rows = []
for m in metrics:
    x = agg['possession_pct'].values
    y = agg[m].values
    mask = (~np.isnan(x)) & (~np.isnan(y))
    n = int(mask.sum())
    if n >= 3:
        pr, pp = stats.pearsonr(x[mask], y[mask])
        sr = stats.spearmanr(x[mask], y[mask])
        rows.append({'metric':m, 'n':n, 'pearson_r':pr, 'pearson_p':pp,
                     'spearman_r': sr.correlation, 'spearman_p': sr.pvalue})
    else:
        rows.append({'metric':m, 'n':n, 'pearson_r':np.nan, 'pearson_p':np.nan,
                     'spearman_r':np.nan, 'spearman_p':np.nan})
corr_table = pd.DataFrame(rows)[['metric','n','pearson_r','pearson_p','spearman_r','spearman_p']]
display(corr_table)

# --- 4) plots & interpretation ---
def interpret_r(r):
    if np.isnan(r): return "tidak cukup data"
    ar = abs(r)
    if ar < 0.1: return "sangat lemah / hampir tidak ada korelasi"
    if ar < 0.3: return "lemah"
    if ar < 0.5: return "sedang"
    return "kuat"

for m in metrics:
    plt.figure(figsize=(6,4))
    sns.regplot(data=agg, x='possession_pct', y=m, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    row = corr_table[corr_table['metric']==m].iloc[0]
    r = row['pearson_r'] if not np.isnan(row['pearson_r']) else 0.0
    p = row['pearson_p'] if not np.isnan(row['pearson_p']) else np.nan
    plt.title(f'Possession vs {m}  (pearson r={r:.2f} p={p:.3g})')
    plt.xlabel('Possession (%)')
    plt.ylabel(m)
    plt.tight_layout()
    plt.show()

print("Ringkasan interpretasi (rule-of-thumb):")
for _, row in corr_table.iterrows():
    print(f"- {row['metric']}: n={int(row['n'])}, pearson r={row['pearson_r']:.3f}, p={row['pearson_p']:.3g} -> {interpret_r(row['pearson_r'])}")

# %% [markdown]
# #### 6. Siapa pemain yang paling sering menjadi "creator" sebelum shot (pass langsung sebelum shot)?

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# safe copies
ev = events_data.copy() if 'events_data' in globals() else None
md = match_data.copy() if 'match_data' in globals() else None

if ev is None or md is None:
    print("events_data atau match_data tidak tersedia — jalankan sel load dataset terlebih dahulu.")
else:
    # Normalize index / ordering (assume dataframe order is chronological if no timestamp)
    ev = ev.reset_index(drop=False).rename(columns={'index': 'orig_index'})
    ev = ev.sort_values(['match_id','orig_index']).reset_index(drop=True)

    # Detect columns
    team_col = next((c for c in ['team.name','possession_team.name','team'] if c in ev.columns), None)
    player_col = next((c for c in ['player.name','player'] if c in ev.columns), None)
    type_col = next((c for c in ['type.name','event.type','type'] if c in ev.columns), None)
    shot_outcome_col = next((c for c in ['shot.outcome.name','shot.outcome','type.name'] if c in ev.columns), None)
    pass_shot_assist_col = next((c for c in ['pass.shot_assist','pass.shot_assist.name','pass.shot_assist_flag'] if c in ev.columns), None)

    def is_shot_row(r):
        # robust shot detection
        if type_col and isinstance(r.get(type_col), str) and 'shot' in r.get(type_col).lower():
            return True
        if shot_outcome_col and isinstance(r.get(shot_outcome_col), str) and any(k in r.get(shot_outcome_col).lower() for k in ('goal','on target','saved','shot')):
            return True
        return False

    def is_pass_row(r):
        if type_col and isinstance(r.get(type_col), str) and 'pass' in r.get(type_col).lower():
            return True
        # fallback: presence of pass.recipient.name or pass.outcome.name
        if 'pass.recipient.name' in ev.columns and pd.notna(r.get('pass.recipient.name')):
            return True
        if 'pass.outcome.name' in ev.columns and pd.notna(r.get('pass.outcome.name')):
            return True
        return False

    ev['_is_shot'] = ev.apply(is_shot_row, axis=1)
    ev['_is_pass'] = ev.apply(is_pass_row, axis=1)

    # Look-back N events to find passer before each shot (same match, prefer same team)
    N_LOOKBACK = 3
    creators_rows = []
    for match_id, g in ev.groupby('match_id', sort=False):
        g = g.reset_index(drop=True)
        for i, row in g[g['_is_shot']].iterrows():
            shooter = row.get(player_col, None)
            shooter_team = row.get(team_col, None)
            shot_idx = row.name
            shot_goal = 0
            if shot_outcome_col and pd.notna(row.get(shot_outcome_col)):
                try:
                    shot_goal = int(str(row.get(shot_outcome_col)).lower().count('goal')>0)
                except:
                    shot_goal = 0
            # search back up to N_LOOKBACK (only within same match order)
            passer = None
            passer_team = None
            pass_type = None
            pass_shot_assist = 0
            for j in range(max(0, i - N_LOOKBACK), i)[::-1]:
                prev = g.iloc[j]
                if prev['_is_pass']:
                    # require same team for creator
                    if pd.notna(prev.get(team_col)) and pd.notna(shooter_team) and str(prev.get(team_col)).strip().lower() == str(shooter_team).strip().lower():
                        passer = prev.get(player_col, None)
                        passer_team = prev.get(team_col, None)
                        # pass type fields if available
                        for c in ['pass.type.name','pass.height.name','pass.body_part.name','pass.recipient.name']:
                            if c in prev.index and pd.notna(prev.get(c)):
                                pass_type = prev.get(c)
                                break
                        if pass_shot_assist_col and pass_shot_assist_col in prev.index:
                            try:
                                pass_shot_assist = int(bool(prev.get(pass_shot_assist_col)))
                            except:
                                pass_shot_assist = 0
                        break
            creators_rows.append({
                'match_id': match_id,
                'shot_index': row['orig_index'],
                'shooter': shooter,
                'shooter_team': shooter_team,
                'passer': passer,
                'passer_team': passer_team,
                'pass_type': pass_type,
                'pass_shot_assist': pass_shot_assist,
                'is_goal': shot_goal
            })

    creators_df = pd.DataFrame(creators_rows)
    # drop shots without identified passer
    creators_with_pass = creators_df[creators_df['passer'].notna()].copy()

    if creators_with_pass.empty:
        print("Tidak ditemukan pass langsung sebelum shot (per konfigurasi). Cek nama kolom atau naikkan N_LOOKBACK.")
    else:
        # Aggregate overall creators
        agg = creators_with_pass.groupby('passer').agg(
            creations=('passer', 'count'),
            creations_goal=('is_goal', 'sum'),
            teams=('passer_team', lambda x: x.mode().iloc[0] if len(x.mode())>0 else np.nan)
        ).reset_index().sort_values('creations', ascending=False)

        # Add conversion and xG per creation
        agg['goal_rate'] = agg['creations_goal'] / agg['creations']

        display(agg.head(30))

        # Top creators plot
        topn = 15
        plt.figure(figsize=(10,5))
        sns.barplot(data=agg.head(topn), x='creations', y='passer', palette='viridis')
        plt.title(f'Top {topn} Creators (pass immediately before shot)')
        plt.xlabel('Creations (count)')
        plt.ylabel('Passer')
        plt.tight_layout()
        plt.show()

        # Barcelona-specific creators (if team names present)
        if creators_with_pass['passer_team'].notna().any():
            is_barca = creators_with_pass['passer_team'].astype(str).str.contains('barcelona', case=False, na=False)
            barca_creators = creators_with_pass[is_barca].copy()
            if not barca_creators.empty:
                agg_barca = barca_creators.groupby('passer').agg(
                    creations=('passer','count'),
                    goals=('is_goal','sum'),
                ).reset_index().sort_values('creations', ascending=False)
                agg_barca['goal_rate'] = agg_barca['goals'] / agg_barca['creations']
                display(agg_barca.head(20))
                plt.figure(figsize=(8,4))
                sns.barplot(data=agg_barca.head(12), x='creations', y='passer', palette='rocket')
                plt.title('Top creators for Barcelona')
                plt.tight_layout()
                plt.show()
            else:
                print("Tidak ada creators dengan passer_team mengandung 'barcelona'.")

        # Per-season (optional) — merge match -> season if match_date/season exists
        if 'match_date' in md.columns or 'season' in md.columns:
            md2 = md[['match_id'] + (['season'] if 'season' in md.columns else []) + (['match_date'] if 'match_date' in md.columns else [])].copy()
            if 'match_date' in md2.columns and 'season' not in md2.columns:
                md2['match_date'] = pd.to_datetime(md2['match_date'], errors='coerce')
                md2['season'] = md2['match_date'].dt.year
            creators_season = creators_with_pass.merge(md2[['match_id','season']], on='match_id', how='left')
            season_table = creators_season.groupby(['season','passer']).agg(ct=('passer','count')).reset_index()
            # show top passer per season
            top_per_season = season_table.sort_values(['season','ct'], ascending=[True,False]).groupby('season').head(5)
            display(top_per_season)


# %% [markdown]
# ### Preprocessing & Feature Engineering

# %%
df_merged.head(50)


# %%
def show_boxplots(df):
    num_cols = df.select_dtypes(include='number').columns
    n = len(num_cols)

    # Hitung baris & kolom grid
    ncols = 3  # jumlah kolom subplot
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')

    # Hapus axis kosong
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

show_boxplots(df_merged)

# %%
df_merged.head(10)

# drop result


# %%
def count_outliers(df):
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_counts[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    return outlier_counts

outliers = count_outliers(df_merged)
outliers_df = pd.DataFrame.from_dict(outliers, orient='index', columns=['Outlier Count'])
outliers_df = outliers_df.sort_values(by='Outlier Count', ascending=False)

print(outliers_df)


# %%
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif

# %%
drop_patterns = [
    r'\bplayer\b', r'\bmanagers\b', r'\breferee\b', r'\bstadium\b', r'\bvenue\b',
    r'name', r'\bcoach\b', r'\barb\b', r'\bcountry\b', r'\bcity\b','kick_off',
    'year','goals_diff','goals_away','barac_goal_diff','shot.outcome.goal.barca',
    'is_home_home','is_away_home','goals_home','home_team.managers_country','away_team.managers_country',
    'home_team.managers_dob','away_team.managers_dob'
]
protected_keep = {'match_id'}  # tetap simpan buat join/submit kalau ada

cols_to_drop = []
for c in df_merged.columns:
    if c in protected_keep:
        continue
    # buang semua object berbau nama/string panjang
    if df_merged[c].dtype == 'O':
        if any(re.search(pat, c, re.I) for pat in drop_patterns):
            cols_to_drop.append(c)

# drop obvious datetime (optional, jika ada)

time_like = [c for c in df_merged.columns if re.search(r'date|time', c, re.I)]
cols_to_drop += time_like




df_cln = df_merged.drop(columns=list(set(cols_to_drop)), errors='ignore').copy()
leak_cols = [c for c in df_cln.columns if re.search(r'(score)', c, re.I)]

df_cln = df_cln.drop(columns=leak_cols, errors='ignore')
df_cln = df_cln.dropna(axis=1, how='all')  # drop all-NaN columns
df_cln = df_cln.reset_index(drop=True)

df_cln = drop_cols_by_names(df_cln, ['match_week','Poss','goals_home','goals_home','is_home_home','is_away_home',
                                     'goals_away','is_home_away','is_away_away','shot.outcome.goal.barca','shot.outcome.goal.nonbarca','goals_diff','barca_goal_diff','year',
                                     ])
test_merged =  drop_cols_by_names(df_cln, ['match_week','Poss','goals_home','goals_home','is_home_home','is_away_home',
                                     'goals_away','is_home_away','is_away_away','shot.outcome.goal.barca','shot.outcome.goal.nonbarca','goals_diff','barca_goal_diff','year',
                                     ])

df_cln.head(10)





# %%
# buat correlation matrix heatmap
plt.figure(figsize=(12,10))
collumns = [c for c in df_cln.select_dtypes(include=[np.number]).columns if df_cln[c].nunique() > 1]
corr = df_cln[collumns].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif

# %%
df_model = df_cln.copy()

# %%
le_target = LabelEncoder()
df_model['Result_encoded'] = le_target.fit_transform(df_model['Result'])

# %%
numeric_features = df_model.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['Result_encoded', 'match_id']
numeric_features = [col for col in numeric_features if col not in exclude_cols]

df_model[numeric_features] = df_model[numeric_features].fillna(df_model[numeric_features].median())
df_model.tail(10)



# %%
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_temp = df_model[numeric_features].copy()
selector.fit(X_temp)
high_var_features = X_temp.columns[selector.get_support()].tolist()

X_fs = df_model[high_var_features].copy()
y_fs = df_model['Result_encoded'].copy()

mi_scores = mutual_info_classif(X_fs, y_fs, random_state=42)
mi_df = pd.DataFrame({
    'feature': high_var_features,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop 20 Features by Mutual Information:")
print(mi_df.head(20))



# %%
K_BEST = min(30, len(high_var_features))  # Pilih top 30 atau semua jika kurang
selected_features = mi_df.head(K_BEST)['feature'].tolist()

print(f"\n✅ Selected {len(selected_features)} features for modeling")

# %%
X = df_model[selected_features].copy()
y = df_model['Result_encoded'].copy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"\nTrain set distribution:\n{pd.Series(y_train).value_counts()}")
print(f"\nTest set distribution:\n{pd.Series(y_test).value_counts()}")

# Scaling dengan RobustScaler (lebih robust terhadap outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        random_state=42,
        eval_metric='mlogloss'
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
}

# %%
results = []
for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print('='*60)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate F1-Macro scores
    f1_train = f1_score(y_train, y_pred_train, average='macro')
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                cv=5, scoring='f1_macro', n_jobs=-1)
    
    results.append({
        'Model': name,
        'F1-Macro (Train)': f1_train,
        'F1-Macro (Test)': f1_test,
        'CV F1-Macro (Mean)': cv_scores.mean(),
        'CV F1-Macro (Std)': cv_scores.std()
    })
    
    print(f"\nF1-Macro Train: {f1_train:.4f}")
    print(f"F1-Macro Test: {f1_test:.4f}")
    print(f"CV F1-Macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Classification report
    print(f"\n{name} - Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, 
                                target_names=le_target.classes_))

# Display results summary
results_df = pd.DataFrame(results).sort_values('F1-Macro (Test)', ascending=False)
print("\n" + "="*80)
print("MODEL COMPARISON - F1-MACRO SCORES")
print("="*80)
print(results_df.to_string(index=False))

# %%
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n🏆 Best Single Model: {best_model_name}")
print(f"F1-Macro (Test): {results_df.iloc[0]['F1-Macro (Test)']:.4f}")

# Create Voting Ensemble of top 3 models
top_3_models = [(name, models[name]) for name in results_df.head(3)['Model'].tolist()]

voting_clf = VotingClassifier(
    estimators=top_3_models,
    voting='soft',
    n_jobs=-1
)

print("\n" + "="*60)
print("Training Voting Ensemble (Top 3 Models)...")
print("="*60)

voting_clf.fit(X_train_scaled, y_train)
y_pred_ensemble = voting_clf.predict(X_test_scaled)

f1_ensemble = f1_score(y_test, y_pred_ensemble, average='macro')
print(f"\n✨ Ensemble F1-Macro (Test): {f1_ensemble:.4f}")

print("\nEnsemble - Classification Report (Test Set):")
print(classification_report(y_test, y_pred_ensemble, 
                           target_names=le_target.classes_))



# %%
best_tree_model = None
for name in ['LightGBM', 'XGBoost', 'Random Forest', 'Gradient Boosting']:
    if name in models:
        best_tree_model = models[name]
        best_tree_name = name
        break

if best_tree_model and hasattr(best_tree_model, 'feature_importances_'):
    importances = best_tree_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 20 Feature Importances ({best_tree_name}):")
    print(feature_imp_df.head(20))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_n = 20
    sns.barplot(data=feature_imp_df.head(top_n), 
                y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances - {best_tree_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

# %%
cm = confusion_matrix(y_test, y_pred_ensemble)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_target.classes_, 
            yticklabels=le_target.classes_)
plt.title(f'Confusion Matrix - Ensemble Model\nF1-Macro: {f1_ensemble:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# %%
print("\n" + "="*60)
print("Hyperparameter Tuning for Best Model...")
print("="*60)

param_grid = {
    'n_estimators': [150, 200, 250],
    'learning_rate': [0.03, 0.05, 0.07],
    'max_depth': [4, 5, 6],
    'num_leaves': [25, 31, 40],
    'min_child_samples': [15, 20, 25]
}

lgbm_tuned = lgb.LGBMClassifier(
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

grid_search = GridSearchCV(
    lgbm_tuned,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1-Macro: {grid_search.best_score_:.4f}")

# Evaluate tuned model
y_pred_tuned = grid_search.best_estimator_.predict(X_test_scaled)
f1_tuned = f1_score(y_test, y_pred_tuned, average='macro')

print(f"Tuned Model F1-Macro (Test): {f1_tuned:.4f}")

print("\nTuned Model - Classification Report:")
print(classification_report(y_test, y_pred_tuned, 
                           target_names=le_target.classes_))

# %%
final_model = grid_search.best_estimator_ if f1_tuned > f1_ensemble else voting_clf
final_f1 = max(f1_tuned, f1_ensemble)
final_model_name = "Tuned LightGBM" if f1_tuned > f1_ensemble else "Voting Ensemble"

print("\n" + "="*80)
print("FINAL MODEL SELECTION")
print("="*80)
print(f"Selected Model: {final_model_name}")
print(f"F1-Macro Score: {final_f1:.4f}")

# %%
import pickle

# Save model
with open('final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le_target, f)

# Save feature list
with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

print("\n✅ Model artifacts saved:")
print("   - final_model.pkl")
print("   - scaler.pkl")
print("   - label_encoder.pkl")
print("   - selected_features.pkl")

# %%
test_merged.to_csv('test_data_processed.csv', index=False)

# %%

def generate_submission(test_data_path, output_path='submission.csv'):
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Load saved artifacts
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('selected_features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    # Prepare test features (same preprocessing as training)
    X_test_sub = test_df[features].copy()
    X_test_sub = X_test_sub.fillna(X_test_sub.median())
    X_test_scaled = scaler.transform(X_test_sub)
    
    # Generate predictions
    predictions = model.predict(X_test_scaled)
    predictions_labels = le.inverse_transform(predictions)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'match_id': test_df['match_id'],
        'Result': predictions_labels
    })
    
    # Save submission
    submission.to_csv(output_path, index=False)
    print(f"\n✅ Submission file saved to: {output_path}")
    print(f"   Shape: {submission.shape}")
    print(f"\nPrediction distribution:")
    print(submission['Result'].value_counts())
    
    return submission
generate_submission('test_data_processed.csv', 'final_submission.csv')


