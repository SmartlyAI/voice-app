# version 3
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
import csv
import os
import uuid
from datetime import datetime
from functools import wraps
import pandas as pd
from pydub import AudioSegment
import io
from huggingface_hub import HfApi, HfFolder, hf_hub_download
import logging
import shutil
from dotenv import load_dotenv

load_dotenv() 
app = Flask(__name__)
app.secret_key = 'une_cle_secrete_tres_securisee_pour_les_sessions_utilisateurs_abc123'
logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION (INCHANGÉE) ---
DATA_DIR = 'data'
RECORDINGS_AUDIO_FOLDERS = {
    "tmz": 'static/audios-tmz-final',
    "darija": 'static/audios-darija-final' 
}
OUTPUT_FILES = {"tmz": os.path.join(DATA_DIR, 'final_dataset-tmz.csv'), "darija": os.path.join(DATA_DIR, 'final_darija_dataset.csv')}
OUTPUT_FIELDNAMES = {
    "tmz": ['user_id', 'nom', 'prenom', 'age', 'genre', 'langue', 'tifinagh', 'latin', 'arabe', 'audio', 'duration_sec', 'timestamp'],
    "darija": ['user_id', 'nom', 'prenom', 'age', 'genre', 'langue', 'latin', 'arabe', 'audio', 'duration_sec', 'timestamp']
}
ADMIN_USERNAME = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASS', 'password123')
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_IDS = { "tmz": "Datasmartly/audio_tamazight_interface", "darija": "Datasmartly/audio_darija_interface" }
HF_DARIJA_SOURCE_REPO = ("Datasmartly/dataset8min", "Datasmartly/dataset-darija-hicham-metadata", "Datasmartly/darija_arabe_only")
HF_TMZ_SOURCE_REPOS = ("Datasmartly/audios-tamazight", "Datasmartly/Tamazight-Mega-Corpus")
SENTENCES_CACHE = {"tmz": [], "darija": []}

# --- FONCTIONS DE CHARGEMENT (INCHANGÉES) ---
def load_sentences_from_hf_csv(repo_id, filename=None):
    try:
        # Détermine le nom du fichier à chercher
        target_filename = filename if filename else (
            "phrases-darija.csv" if "darija-hicham" in repo_id 
            else "metadata.csv" if repo_id == "Datasmartly/dataset8min"
            else "darija_arabe_only.csv" 
        )
        
        app.logger.info(f"TÉLÉCHARGEMENT du fichier '{target_filename}' depuis {repo_id}...")
        csv_path = hf_hub_download(repo_id=repo_id, filename=target_filename, repo_type="dataset")
        
        sentences = []
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Gère les deux formats de fichiers
                if "darija-hicham" in repo_id:
                    sentences.append({
                        'latin': row.get('darija_ltn', '').strip(),
                        'arabe': row.get('darija_ar', '').strip(),
                        'audio_filename': None,  # Ce dataset n'a pas d'audio
                        'source_repo': repo_id
                    })
                elif repo_id == "Datasmartly/darija_arabe_only":
                    sentences.append({
                        'latin': '',  # Colonne vide
                        'arabe': row.get('darija_arabe', '').strip(),
                        'audio_filename': None,
                        'source_repo': repo_id
                    })
                else:  # Pour dataset8min
                    sentences.append({
                        'latin': row.get('transcription_darija_ltn', '').strip(),
                        'arabe': row.get('transcription_darija_ar', '').strip(),
                        'audio_filename': row.get('file_name'),
                        'source_repo': repo_id
                    })
        
        app.logger.info(f"{len(sentences)} phrases Darija chargées depuis {repo_id}")
        return sentences
        
    except Exception as e:
        app.logger.error(f"Erreur avec {repo_id}: {e}", exc_info=True)
        return []
    
def load_tamazight_with_audio_from_hf(repo_id):
    app.logger.info(f"Chargement des données Tamazight AVEC AUDIO depuis le dépôt HF : {repo_id}")
    try:
        downloaded_path = hf_hub_download(repo_id=repo_id, filename="data/Transcriptions.csv", repo_type="dataset")
        df = pd.read_csv(downloaded_path, header=0, names=['audio_filename', 'tifinagh']); df.dropna(inplace=True); df = df[df['tifinagh'].str.strip() != '']
        sentences = [{'tifinagh': row['tifinagh'], 'latin': '', 'arabe': '', 'audio_filename': row['audio_filename'], 'source_repo': repo_id} for _, row in df.iterrows()]
        app.logger.info(f"Chargement Tamazight avec audio terminé. {len(sentences)} phrases chargées depuis {repo_id}.")
        return sentences
    except Exception as e:
        app.logger.error(f"Erreur lors du traitement du dépôt {repo_id}: {e}", exc_info=True)
        return []
def load_tamazight_text_only_from_hf(repo_id):
    app.logger.info(f"Chargement des données Tamazight (texte seul) depuis le dépôt HF : {repo_id}")
    all_sentences = []
    try:
        repo_files = [f for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset") if '/' in f and f.endswith('.csv') and '.gitattributes' not in f]
        app.logger.info(f"Trouvé {len(repo_files)} fichiers CSV à traiter dans {repo_id}.")
        for csv_file_path in repo_files:
            try:
                downloaded_path = hf_hub_download(repo_id=repo_id, filename=csv_file_path, repo_type="dataset")
                df = pd.read_csv(downloaded_path).rename(columns={'Tamazight_Tifinagh': 'tifinagh', 'Tamazight_Arabic': 'arabe', 'Tamazight_Latin': 'latin'})
                if not all(col in df.columns for col in ['tifinagh', 'arabe', 'latin']): continue
                subset_df = df[['tifinagh', 'arabe', 'latin']].fillna(''); subset_df['audio_filename'] = None; subset_df['source_repo'] = repo_id
                all_sentences.extend(subset_df.to_dict('records'))
            except Exception as file_e:
                app.logger.error(f"    -> Erreur lors du traitement du fichier {csv_file_path}: {file_e}")
        app.logger.info(f"Chargement Tamazight (texte seul) terminé. Total de {len(all_sentences)} phrases chargées.")
        return all_sentences
    except Exception as e:
        app.logger.error(f"Erreur critique lors de l'accès au dépôt {repo_id}: {e}", exc_info=True)
        return []

def populate_sentences_cache():
    app.logger.info("--- DÉBUT DU PRÉ-CHARGEMENT DES DONNÉES ---")
    
    # Chargement Darija (multi-sources)
    darija_sentences = []
    for repo_id in HF_DARIJA_SOURCE_REPO:
        sentences = load_sentences_from_hf_csv(repo_id)
        darija_sentences.extend(sentences)
    SENTENCES_CACHE["darija"] = darija_sentences
    
    # Chargement Tamazight (inchangé)
    tmz_sentences = []
    tmz_sentences.extend(load_tamazight_with_audio_from_hf(HF_TMZ_SOURCE_REPOS[0]))
    tmz_sentences.extend(load_tamazight_text_only_from_hf(HF_TMZ_SOURCE_REPOS[1]))
    SENTENCES_CACHE["tmz"] = tmz_sentences
    
    app.logger.info("--- CACHE PRÊT ---")
    app.logger.info(f"TMZ: {len(SENTENCES_CACHE['tmz'])} phrases | Darija: {len(SENTENCES_CACHE['darija'])} phrases")


def load_sentences(langue):
    return SENTENCES_CACHE.get(langue, [])

# --- INITIALISATION AU DÉMARRAGE (INCHANGÉE) ---
HfFolder.save_token(HF_TOKEN)
api = HfApi()
try:
    user_info = api.whoami()
    app.logger.info(f"Hugging Face token valide. Connecté en tant que : {user_info['name']}")
except Exception as e:
    app.logger.warning(f"Avertissement : Impossible de valider le token Hugging Face. L'envoi échouera probablement. Erreur : {e}")
for folder in RECORDINGS_AUDIO_FOLDERS.values(): os.makedirs(folder, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
for lang, path in OUTPUT_FILES.items():
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES[lang])
            writer.writeheader()
populate_sentences_cache()

# --- FONCTIONS UTILITAIRES (INCHANGÉES) ---
def push_to_huggingface(langue, audio_object_in_memory, audio_filename, csv_content_in_memory):
    repo_id = HF_REPO_IDS.get(langue)
    if not repo_id: return
    try:
        api.upload_file(path_or_fileobj=audio_object_in_memory, path_in_repo=f"audio/{audio_filename}", repo_id=repo_id, repo_type="dataset")
        app.logger.info(f"Upload de l'audio {audio_filename} réussi.")
        csv_bytes = csv_content_in_memory.encode('utf-8')
        api.upload_file(path_or_fileobj=csv_bytes, path_in_repo="metadata.csv", repo_id=repo_id, repo_type="dataset")
        app.logger.info(f"Upload du metadata.csv réussi pour l'enregistrement {audio_filename}.")
    except Exception as e:
        app.logger.error(f"Échec de l'upload vers Hugging Face pour {audio_filename}: {e}", exc_info=True)
        raise e

def add_entry_to_dataset(user_id, user_info, langue, phrase_data, audio_filename, duration):
    output_file = OUTPUT_FILES[langue]
    fieldnames = OUTPUT_FIELDNAMES[langue]
    file_exists = os.path.isfile(output_file) and os.path.getsize(output_file) > 0
    new_entry = { 'user_id': user_id, 'nom': user_info['nom'], 'prenom': user_info['prenom'], 'age': user_info.get('age', 'N/A'), 'genre': user_info['genre'], 'langue': langue, 'tifinagh': phrase_data.get('tifinagh', ''), 'latin': phrase_data.get('latin', ''), 'arabe': phrase_data.get('arabe', ''), 'audio': f"audio/{audio_filename}", 'duration_sec': str(duration), 'timestamp': datetime.now().isoformat() }
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists: writer.writeheader()
        writer.writerow(new_entry)

# ==========================================================
# DÉBUT DE LA MODIFICATION 1/2
# ==========================================================
def get_user_recorded_data(user_id, langue):
    output_file = OUTPUT_FILES.get(langue)
    # MODIFIÉ : On va retourner l'identifiant de la dernière phrase, pas juste le latin
    last_recorded_identifier = None 
    recorded_identifiers_set = set()

    if not os.path.isfile(output_file):
        return last_recorded_identifier, recorded_identifiers_set
        
    try:
        df = pd.read_csv(output_file, on_bad_lines='skip', low_memory=False)
        if df.empty or 'user_id' not in df.columns:
            return last_recorded_identifier, recorded_identifiers_set

        df['user_id'] = df['user_id'].astype(str)
        user_recordings = df[df['user_id'] == user_id]

        if user_recordings.empty:
            return last_recorded_identifier, recorded_identifiers_set

        def create_identifier(row):
            tifinagh = str(row.get('tifinagh', '')) if pd.notna(row.get('tifinagh')) else ''
            latin = str(row.get('latin', '')) if pd.notna(row.get('latin')) else ''
            arabe = str(row.get('arabe', '')) if pd.notna(row.get('arabe')) else ''
            return f"{tifinagh}-{latin}-{arabe}"
        
        recorded_identifiers_set = set(user_recordings.apply(create_identifier, axis=1))

        # Trouver la dernière phrase enregistrée en se basant sur le timestamp
        user_recordings_copy = user_recordings.copy()
        user_recordings_copy['timestamp'] = pd.to_datetime(user_recordings_copy['timestamp'], errors='coerce')
        user_recordings_copy.dropna(subset=['timestamp'], inplace=True)
        if not user_recordings_copy.empty:
            last_recording = user_recordings_copy.sort_values(by='timestamp', ascending=False).iloc[0]
            # MODIFIÉ : On génère et stocke l'identifiant complet de la dernière phrase
            last_recorded_identifier = create_identifier(last_recording)

    except Exception as e:
        app.logger.error(f"Error processing user data from {output_file}: {e}", exc_info=True)
        
    return last_recorded_identifier, recorded_identifiers_set
# ==========================================================
# FIN DE LA MODIFICATION 1/2
# ==========================================================


# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prenom = request.form['prenom'].lower().strip(); nom = request.form['nom'].lower().strip()
        session['user'] = {'nom': nom, 'prenom': prenom, 'age': request.form.get('age'), 'genre': request.form['genre']}
        session['user_id'] = f"{prenom}_{nom}"; session['langue'] = request.form['langue']; session.pop('current_sentence_index', None)
        return redirect(url_for('recorder'))
    if 'user' in session: return redirect(url_for('recorder'))
    return render_template('index.html')

@app.route('/recorder')
def recorder():
    if 'user' not in session: return redirect(url_for('index'))
    langue, user_id = session['langue'], session['user_id']; all_sentences = load_sentences(langue)
    if not all_sentences:
        flash(f"Impossible de charger les phrases pour la langue {langue}. Veuillez contacter un administrateur.", "danger")
        return render_template('merci.html', message=f"Problème de chargement des données pour la langue '{langue}'.")

    # ==========================================================
    # DÉBUT DE LA MODIFICATION 2/2
    # ==========================================================
    # MODIFIÉ : On récupère l'identifiant de la dernière phrase, et non plus 'last_latin'
    last_recorded_identifier, recorded_identifiers_set = get_user_recorded_data(user_id, langue)
    
    current_idx_from_session = session.get('current_sentence_index'); target_idx = 0
    
    # Si c'est la première visite (pas d'index dans la session)
    if current_idx_from_session is None:
        # MODIFIÉ : On vérifie si on a trouvé un identifiant de dernière phrase
        if last_recorded_identifier:
            try:
                # MODIFIÉ : On cherche l'index en comparant les identifiants complets
                last_idx = next(i for i, s in enumerate(all_sentences) 
                                if f"{s.get('tifinagh', '')}-{s.get('latin', '')}-{s.get('arabe', '')}" == last_recorded_identifier)
                # On se positionne sur la phrase suivante
                target_idx = last_idx + 1 if last_idx < len(all_sentences) - 1 else len(all_sentences)
            except StopIteration:
                # Si la dernière phrase enregistrée n'est plus dans la liste, on repart du début
                target_idx = 0
    else:
        # Si on navigue avec Précédent/Suivant, on utilise l'index de la session
        target_idx = int(current_idx_from_session)
    # ==========================================================
    # FIN DE LA MODIFICATION 2/2
    # ==========================================================
    
    if target_idx >= len(all_sentences): return render_template('merci.html', message="Félicitations ! Vous avez terminé toutes les phrases.")
    session['current_sentence_index'] = target_idx; phrase_to_display = all_sentences[target_idx]

    # La vérification du statut "enregistré" utilise déjà la bonne méthode (identifiant)
    phrase_identifier = f"{phrase_to_display.get('tifinagh', '')}-{phrase_to_display.get('latin', '')}-{phrase_to_display.get('arabe', '')}"
    is_phrase_recorded_by_user = phrase_identifier in recorded_identifiers_set

    reference_audio_url = None; audio_filename = phrase_to_display.get('audio_filename')
    if audio_filename:
        source_repo = phrase_to_display.get('source_repo')
        if source_repo:
            if source_repo == "Datasmartly/audios-tamazight":
                # Le chemin est `data/nom_fichier.mp3` dans ce repo spécifique
                reference_audio_url = f"https://huggingface.co/datasets/{source_repo}/resolve/main/data/audios/{audio_filename}"
            else: 
                # Le chemin est `data/audios/nom_fichier.mp3` pour les autres
                reference_audio_url = f"https://huggingface.co/datasets/{source_repo}/resolve/main/data/audios/{audio_filename}"
                
    return render_template('recorder.html', phrase=phrase_to_display, current_phrase_number=target_idx + 1, total_phrases=len(all_sentences), is_phrase_recorded=is_phrase_recorded_by_user, reference_audio_url=reference_audio_url, upload_url=url_for('upload'))

@app.route('/next', methods=['POST'])
def next_phrase():
    if 'user' not in session: return redirect(url_for('index'))
    total_phrases = len(load_sentences(session['langue'])); current_index = session.get('current_sentence_index', 0)
    if current_index < total_phrases - 1: session['current_sentence_index'] = current_index + 1
    return redirect(url_for('recorder'))

@app.route('/previous', methods=['POST'])
def previous_phrase():
    if 'user' not in session: return redirect(url_for('index'))
    current_index = session.get('current_sentence_index', 0)
    if current_index > 0: session['current_sentence_index'] = current_index - 1
    return redirect(url_for('recorder'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session: return jsonify({'success': False, 'message': 'Non autorisé'}), 401
    
    user_info, user_id, langue = session['user'], session['user_id'], session['langue']
    file = request.files.get('audio_data')
    if not file: return jsonify({'success': False, 'message': 'Aucun fichier audio'}), 400
    
    duration_float = float(request.form.get('duration', '0'))
    phrase_data = {'tifinagh': request.form.get('tifinagh', ''), 'latin': request.form.get('latin', ''), 'arabe': request.form.get('arabe', '')}
    if not any(v.strip() for v in phrase_data.values()): 
        return jsonify({'success': False, 'message': 'Données de texte manquantes'}), 400
    
    final_filename = f"audio_{user_id}_{langue}_{uuid.uuid4().hex[:8]}.wav"
    
    try:
        sound = AudioSegment.from_file(io.BytesIO(file.stream.read()))
        wav_buffer = io.BytesIO()
        sound.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        output_file_path = OUTPUT_FILES[langue]
        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
            df = pd.read_csv(output_file_path, on_bad_lines='skip')
        else:
            df = pd.DataFrame(columns=OUTPUT_FIELDNAMES[langue])

        new_entry = { 'user_id': user_id, 'nom': user_info['nom'], 'prenom': user_info['prenom'], 'age': user_info.get('age', 'N/A'), 'genre': user_info['genre'], 'langue': langue, 'tifinagh': phrase_data.get('tifinagh', ''), 'latin': phrase_data.get('latin', ''), 'arabe': phrase_data.get('arabe', ''), 'audio': f"audio/{final_filename}", 'duration_sec': str(duration_float), 'timestamp': datetime.now().isoformat() }
        new_df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        csv_content_for_upload = new_df.to_csv(index=False)

        push_to_huggingface(langue, wav_buffer, final_filename, csv_content_for_upload)
        
        app.logger.info("Envoi HF réussi, sauvegarde des métadonnées locales en cours...")
        add_entry_to_dataset(user_id, user_info, langue, phrase_data, final_filename, duration_float)
            
        return jsonify({'success': True, 'message': 'Enregistrement validé et envoyé !'})

    except Exception as e:
        app.logger.error(f"Échec du processus d'upload. Aucune donnée locale n'a été sauvegardée. Erreur: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f"L'envoi a échoué. Veuillez vérifier votre connexion et réessayer."}), 500

@app.route('/logout')
def logout():
    session.clear()
    flash('Vous avez été déconnecté avec succès.', 'info')
    return redirect(url_for('index'))

# --- SECTION ADMIN (INCHANGÉE) ---
def format_duration(seconds_str):
    try: seconds = float(seconds_str)
    except (ValueError, TypeError): return "0s"
    hours, remainder = divmod(seconds, 3600); minutes, secs = divmod(remainder, 60)
    if hours > 0: return f"{int(hours)}h {int(minutes)}m {int(secs)}s"
    if minutes > 0: return f"{int(minutes)}m {int(secs)}s"
    return f"{int(secs)}s"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'): return redirect(url_for('admin_login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form['username'] == ADMIN_USERNAME and request.form['password'] == ADMIN_PASSWORD:
            session['admin_logged_in'] = True; return redirect(request.args.get('next') or url_for('admin_dashboard'))
        flash('Identifiants incorrects.', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None); return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    all_dfs = []
    for lang, filepath in OUTPUT_FILES.items():
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try: df_lang = pd.read_csv(filepath); df_lang.rename(columns={'audio': 'audio_filename'}, inplace=True); all_dfs.append(df_lang)
            except Exception as e: app.logger.error(f"Dashboard: Error reading {filepath}: {e}")
    if not all_dfs:
        flash("Aucune donnée d'enregistrement à afficher.", "warning"); return render_template('admin_dashboard.html', stats={}, format_duration=format_duration)
    df = pd.concat(all_dfs, ignore_index=True)
    stats = {'total_recordings': 0, 'total_duration_sec': 0.0, 'gender_distribution': {}, 'user_stats': {}, 'recordings_per_day_labels': [], 'recordings_per_day_data': [], 'duration_per_user_labels': [], 'duration_per_user_data': [], 'duration_per_user_per_day_labels': [], 'duration_per_user_per_day_datasets': [], 'duration_per_day_data': []}
    try:
        if df.empty:
            flash("Les fichiers de données sont vides.", "warning"); return render_template('admin_dashboard.html', stats=stats, format_duration=format_duration)
        df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce').fillna(0.0); df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce'); df.dropna(subset=['timestamp'], inplace=True); df['full_name'] = df['prenom'].astype(str).str.strip() + ' ' + df['nom'].astype(str).str.strip(); df['full_name'] = df['full_name'].str.strip().replace('', 'Unknown User'); stats['total_recordings'] = len(df); stats['total_duration_sec'] = df['duration_sec'].sum(); gender_counts = df.drop_duplicates(subset=['full_name'])['genre'].fillna('Unknown').value_counts(); stats['gender_distribution'] = (gender_counts / gender_counts.sum() * 100).round(1).to_dict() if not gender_counts.empty else {}
        user_grouped = df.groupby('full_name').agg(record_count=('audio_filename', 'count'), total_duration=('duration_sec', 'sum')).reset_index()
        for _, row in user_grouped.iterrows():
            user_name = row['full_name']
            if not user_name or user_name == 'Unknown User': continue
            stats['user_stats'][user_name] = {'record_count': row['record_count'], 'total_duration': row['total_duration'], 'daily_activity': {}}
            user_df = df[df['full_name'] == user_name].copy()
            if not user_df.empty:
                user_df.loc[:, 'date'] = user_df['timestamp'].dt.date
                daily_grouped = user_df.groupby('date').agg(daily_record_count=('audio_filename', 'count'), daily_total_duration=('duration_sec', 'sum')).reset_index()
                for _, daily_row in daily_grouped.iterrows():
                    stats['user_stats'][user_name]['daily_activity'][daily_row['date'].strftime('%Y-%m-%d')] = {'count': daily_row['daily_record_count'], 'duration': daily_row['daily_total_duration']}
        if not df.empty:
            df_for_daily_stats = df.copy(); df_for_daily_stats.loc[:, 'date_only'] = df_for_daily_stats['timestamp'].dt.date
            recordings_per_day = df_for_daily_stats.groupby('date_only').size().sort_index()
            if not recordings_per_day.empty: stats['recordings_per_day_labels'] = [d.strftime('%Y-%m-%d') for d in recordings_per_day.index]; stats['recordings_per_day_data'] = recordings_per_day.values.tolist()
            duration_per_day = df_for_daily_stats.groupby('date_only')['duration_sec'].sum().sort_index()
            if not duration_per_day.empty: stats['duration_per_day_data'] = [round(v / 60, 2) for v in duration_per_day.values]
        stats['duration_per_user_labels'] = [name for name in stats['user_stats'].keys() if name and name != 'Unknown User']; stats['duration_per_user_data'] = [round(user_data['total_duration'] / 60, 2) for name, user_data in stats['user_stats'].items() if name and name != 'Unknown User']; all_dates = sorted(list({date for user_data in stats['user_stats'].values() for date in user_data.get('daily_activity', {}).keys()})); stats['duration_per_user_per_day_labels'] = all_dates; colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#F67019', '#00C49F']; user_index = 0
        for full_name, user_data in stats['user_stats'].items():
            if not full_name or full_name == 'Unknown User': continue
            data_points = [round(user_data.get('daily_activity', {}).get(date, {}).get('duration', 0) / 60, 2) for date in all_dates]
            stats['duration_per_user_per_day_datasets'].append({'label': full_name, 'data': data_points, 'borderColor': colors[user_index % len(colors)], 'backgroundColor': colors[user_index % len(colors)], 'fill': False, 'tension': 0.1}); user_index += 1
    except Exception as e:
        app.logger.error(f"Dashboard generation error: {e}", exc_info=True)
        flash(f"Erreur lors de la génération des statistiques: {e}", "danger")
    return render_template('admin_dashboard.html', stats=stats, format_duration=format_duration)

@app.route('/jump_to_phrase', methods=['GET'])
def jump_to_phrase():
    if 'user' not in session:
        return redirect(url_for('index'))
    
    phrase_number = request.args.get('phrase_number', type=int)
    langue = session['langue']
    all_sentences = load_sentences(langue)
    
    if not phrase_number or phrase_number < 1 or phrase_number > len(all_sentences):
        flash("Numéro de phrase invalide", "danger")
        return redirect(url_for('recorder'))
    
    session['current_sentence_index'] = phrase_number - 1
    return redirect(url_for('recorder'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)

