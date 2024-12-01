from datetime import time

import pymysql

pymysql.install_as_MySQLdb()
import os
import re
import uuid
import arabic_reshaper
import nltk
import json
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
from matplotlib import rcParams
from sympy.physics.control.control_plots import matplotlib
from werkzeug.security import generate_password_hash, check_password_hash
from collections import Counter
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import MySQLdb.cursors
import mishkal
from camel_tools.utils.charmap import CharMapper
from camel_tools.tokenizers.word import simple_word_tokenize
# from camel_tools.stem.isri import ISRIStemmer
from nltk.stem.isri import ISRIStemmer
import smtplib
from email.mime.text import MIMEText
# New imports for NER and Keyword Extraction
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
import torch
import numpy as np
from keybert import KeyBERT
import uuid
from docx import Document

from flask import Response, stream_with_context

# pip install python-docx


app = Flask(__name__)

app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

# MySQL Configuration
# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
# Default MAMP port for MySQL
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'  # Default MAMP password (if unchanged)
app.config['MYSQL_DB'] = 'flask_users2'
app.config['MYSQL_SSL_DISABLED'] = True  # Disable SSL

# Initialize MySQL
mysql = MySQL(app)
matplotlib.use('Agg')  # Ensure matplotlib doesn't require an active display

# Set up Arabic text rendering in matplotlib
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Amiri']  # Ensure 'Amiri' font is available
rcParams['axes.unicode_minus'] = False

# Download required NLTK packages
nltk.download('punkt')
nltk.download('stopwords')

# Use NLTK's Arabic stopwords list
arabic_stopwords = set(nltk.corpus.stopwords.words("arabic"))

# Manually add missing entries to the stopword list
additional_stopwords = {
    "او", "ان", "انها", "أنه", "انه", "أن", "أنها", "إلى", "الي", "على", "علي",
    "وهي", "ولم", "وان", "الا", "بهذه", "ومع", "ام", "لدي", "فلم", "وانها",
    "فانهم", "وهناك", "لهذه", "وذلك", "وعلي", "اذ", "فان", "لان", "وحتي", "وفي",
    "ولان", "اذا", "وهذا", "وبهذا", "فهذه", "كانت"
}
arabic_stopwords.update(additional_stopwords)

# Initialize NLTK stemmer for Arabic
stemmer = ISRIStemmer()

# Initialize models for NER and Keyword Extraction
model_ner_name = "marefa-nlp/marefa-ner"
tokenizer_ner = AutoTokenizer.from_pretrained(model_ner_name)
model_ner = AutoModelForTokenClassification.from_pretrained(model_ner_name)

model_kw_name = "aubmindlab/bert-base-arabertv02"
tokenizer_kw = AutoTokenizer.from_pretrained(model_kw_name)
model_kw = AutoModel.from_pretrained(model_kw_name)
kw_model = KeyBERT(model_kw)

# Custom labels for NER
custom_labels = [
    "O", "B-job", "I-job", "B-nationality", "B-person", "I-person",
    "B-location", "B-time", "I-time", "B-event", "I-event",
    "B-organization", "I-organization", "I-location", "I-nationality",
    "B-product", "I-product", "B-artwork", "I-artwork"
]

# Path to save uploaded files
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure directory exists

# Initialize character mapper for normalization
mapper = CharMapper.builtin_mapper('arclean')

# List of common Arabic words with valid double letters (without شدّة)
valid_double_letter_words = [
    "مرر", "سدد", "ضرر", "حلل", "قرر", "جرر", "كرر", "فرر", "مدد", "ردد",
    "شدد", "هدد", "خطط", "برر", "عمم", "رمم", "حلل", "سدد", "مدد", "كرّر",
    "هدد", "ضرر", "مرر", "قلل", "خطط", "مدد", "برر", "رمم", "زرر", "جفف",
    "تتلألأ", "اللغه", "اللغة", "تتلألأ"
]

smtp_server = "smtp.gmail.com"
smtp_port = 587


# Authentication and home page routes
@app.route('/index')
def index():
    if 'loggedin' in session:
        return redirect(url_for('upload_file'))
    return redirect(url_for('login'))


from werkzeug.security import check_password_hash


# def send_verification_email(email, verification_token):
#  verification_link = url_for('verify_email', token=verification_token, _external=True)
#  message = f'Please verify your email by clicking the following link: {verification_link}'

#  msg = MIMEText(message)
#  msg['Subject'] = 'Email Verification'
#  msg['From'] = 'bayyinhelp@gmail.com'
#  msg['To'] = email

#   with smtplib.SMTP('smtp.gmail.com', 587) as server:  # Use your SMTP server
#      server.starttls()
#      server.login('bayyinhelp@gmail.com', 'kgrz otqt rckn pnwv')
#    server.sendmail(msg['From'], [msg['To']], msg.as_string())
@app.route('/auth', methods=['POST'])
def auth():
    action = request.form.get('action')
    if action == 'login':
        # Get login credentials from form
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            return render_template('login.html', error_message="يرجى إدخال اسم المستخدم وكلمة المرور.")

        # Query the user from the database
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM tbl_users WHERE username = %s', (username,))
        user = cursor.fetchone()

        if user:
            # Check if the user has verified their email
            if not user['is_verified']:
                return render_template('login.html', error_message="يرجى التحقق من بريدك الإلكتروني قبل تسجيل الدخول.")

            # Verify the password
            if check_password_hash(user['password'], password):
                session['loggedin'] = True
                session['username'] = user['username']
                session['id'] = user['id']
                return redirect(url_for('home'))
            else:
                return render_template('login.html', error_message="اسم المستخدم أو كلمة المرور غير صحيحة!")
        else:
            return render_template('login.html', error_message="اسم المستخدم أو كلمة المرور غير صحيحة!")

    elif action == 'register':
        # Get registration details from form
        first_name = request.form.get('first_name', '').strip()
        second_name = request.form.get('second_name', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        # Validate inputs
        if not all([first_name, second_name, username, email, password]):
            return render_template('login.html', error_message="يرجى ملء جميع الحقول المطلوبة.")

        if not first_name.isalpha() or not second_name.isalpha():
            return render_template('login.html', error_message="يجب ألا يحتوي الاسم الأول أو الثاني على أرقام.")

        if "@" not in email or "." not in email:
            return render_template('login.html', error_message="يرجى إدخال بريد إلكتروني صحيح.")

        # Check if the username is already taken
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM tbl_users WHERE username = %s', (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            return render_template('login.html', error_message="اسم المستخدم موجود بالفعل. يرجى اختيار اسم مستخدم آخر.")

        # Hash the password
        password_hashed = generate_password_hash(password)
        verification_token = str(uuid.uuid4())

        try:
            # Insert the new user into the database with verification status as false
            cursor.execute(
                'INSERT INTO tbl_users (first_name, second_name, username, email, password, verification_token, is_verified) VALUES (%s, %s, %s, %s, %s, %s, %s)',
                (first_name, second_name, username, email, password_hashed, verification_token, False)
            )
            mysql.connection.commit()

            # Send a verification email
            send_verification_email(email, verification_token)
            return render_template('verify_email.html',
                                   message="تم إرسال بريد إلكتروني للتحقق. يرجى التحقق من بريدك الإلكتروني.")
        except Exception as e:
            print(f"Error during registration: {e}")
            return f"Registration failed: {str(e)}"

    else:
        return render_template('login.html', error_message="Invalid action")


from flask import redirect, url_for


@app.route('/verify_email/<token>')
def verify_email(token):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM tbl_users WHERE verification_token = %s', (token,))
    user = cursor.fetchone()

    if user and not user['is_verified']:
        cursor.execute('UPDATE tbl_users SET is_verified = %s WHERE id = %s', (True, user['id']))
        mysql.connection.commit()
        return redirect(url_for('login', success_message="تم التحقق من بريدك الإلكتروني. يمكنك الآن تسجيل الدخول."))
    elif user and user['is_verified']:
        return redirect(url_for('login', error_message="هذا البريد الإلكتروني تم التحقق منه مسبقاً."))
    else:
        return redirect(url_for('login', error_message="رابط التحقق غير صالح."))


@app.route('/login')
def login():
    success_message = request.args.get('success_message', '')
    error_message = request.args.get('error_message', '')
    return render_template('login.html', success_message=success_message, error_message=error_message)


def send_verification_email(email, verification_token):
    verification_link = url_for('verify_email', token=verification_token, _external=True)
    message = f'Please verify your email by clicking the following link: {verification_link}'

    msg = MIMEText(message)
    msg['Subject'] = 'Email Verification'
    msg['From'] = 'bayyinhelp@gmail.com'
    msg['To'] = email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:  # Use your SMTP server
            server.starttls()
            server.login('bayyinhelp@gmail.com', 'kgrz otqt rckn pnwv')
            server.sendmail(msg['From'], [msg['To']], msg.as_string())
    except Exception as e:
        print(f"Error sending email: {e}")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        second_name = request.form['second_name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if all fields are filled
        if not (first_name and second_name and username and email and password):
            return render_template('regestr.html', error_message="يرجى ملء جميع الحقول المطلوبة.")

        # Check email and password pattern validation
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return render_template('regestr.html', error_message="يرجى إدخال بريد إلكتروني صحيح.")
        if not re.match(r"(?=.\d)(?=.[!@#$%^&])[A-Za-z\d!@#$%^&]{10,}", password):
            return render_template('regestr.html',
                                   error_message="يجب أن تحتوي كلمة المرور على 10 أحرف على الأقل، ويجب أن تتضمن رقمًا واحدًا ورمزًا خاصًا.")

        password_hashed = generate_password_hash(password)
        verification_token = str(uuid.uuid4())  # Generate verification token

        try:
            cursor = mysql.connection.cursor()
            cursor.execute(
                'INSERT INTO tbl_users (first_name, second_name, username, email, password, verification_token) VALUES (%s, %s, %s, %s, %s, %s)',
                (first_name, second_name, username, email, password_hashed, verification_token)
            )
            mysql.connection.commit()

            # Send verification email
            send_verification_email(email, verification_token)

            # Show confirmation alert and render the registration page with a success message
            return render_template('regestr.html',
                                   success_message="تم إرسال بريد إلكتروني للتحقق. يرجى النقر عليه للتأكيد .")
        except Exception as e:
            print(f"Error during registration: {e}")
            return f"Registration failed: {str(e)}"

    return render_template('regestr.html')


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    session.pop('id', None)
    return redirect(url_for('login'))


# Home page route
@app.route('/home')
def home():
    return render_template('homeBage.html')


@app.route('/terms')
def terms():
    return render_template('terms.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')


from flask import send_file  # Ensure this import is included

# Upload and process file route
# Upload and process file route
from flask import render_template

import uuid


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'files' not in request.files or not request.files.getlist('files'):
            return render_template('upload.html', message='الرجاء اختيار ملف واحد أو أكثر من نوع .txt أو .docx')

        # Retrieve files and save them temporarily
        files = request.files.getlist('files')
        uploaded_files = []
        for file in files:
            file_extension = file.filename.split('.')[-1].lower()
            if file_extension not in ['txt', 'docx']:
                return render_template('upload.html', message='يرجى اختيار ملفات نصية بامتداد .txt أو .docx فقط')

            # Save the file with a unique filename
            filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(file_path)

        # Store the uploaded files' paths in the session
        session['uploaded_files'] = uploaded_files

        # Redirect to processing options page
        return redirect(url_for('processing_options'))

    return render_template('upload.html')


@app.route('/processing', methods=['GET', 'POST'])
def processing_options():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    if 'uploaded_files' not in session:
        return redirect(url_for('upload_file'))

    if request.method == 'POST':
        # Retrieve user-selected options
        processing_option = request.form.get('processing')
        apply_ner = 'apply_ner' in request.form
        apply_key_extraction = 'apply_key_extraction' in request.form
        top_n = int(request.form.get('top_n', 10))
        freq_order = request.form.get('freq_order', 'most')

        # Store processing options in session
        session['processing_option'] = processing_option
        session['apply_ner'] = apply_ner
        session['apply_key_extraction'] = apply_key_extraction
        session['top_n'] = top_n
        session['freq_order'] = freq_order

        # Redirect to results processing
        return redirect(url_for('process_results'))

    return render_template('processing.html')

from time import sleep


@app.route('/process_results', methods=['GET'])
def process_results():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    if 'uploaded_files' not in session or 'processing_option' not in session:
        return redirect(url_for('upload_file'))

    # Simulate backend processing delay to improve synchronization with front-end loading
    sleep(3)  # Simulating a delay (3 seconds) to align the backend with the progress bar in the frontend

    # Retrieve files and options from session
    uploaded_files = session.get('uploaded_files', [])
    processing_option = session.get('processing_option', 'تنظيف فقط')
    apply_ner = session.get('apply_ner', False)
    apply_key_extraction = session.get('apply_key_extraction', False)
    top_n = session.get('top_n', 10)
    freq_order = session.get('freq_order', 'most')

    # Initialize merged text
    merged_text = ""

    # Load and merge text from the uploaded files
    for file_path in uploaded_files:
        file_extension = file_path.split('.')[-1].lower()

        # Handle .txt files
        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                merged_text += f.read() + "\n"

        # Handle .docx files
        elif file_extension == 'docx':
            document = Document(file_path)
            docx_text = "\n".join([paragraph.text for paragraph in document.paragraphs])
            merged_text += docx_text + "\n"

    # Clean and tokenize merged text
    cleaned_text = clean_arabic_text(merged_text)
    tokens = tokenize_text(cleaned_text)

    # Apply processing options
    if processing_option == "تنظيف + إزالة الكلمات الشائعة":
        tokens = remove_stopwords(tokens)
    elif processing_option == "تنظيف + إزالة الكلمات الشائعة + الجذر":
        tokens = remove_stopwords(tokens)
        tokens = stem_words(tokens)

    # Generate word frequencies
    word_frequencies = generate_word_frequencies(tokens)
    sorted_word_frequencies = sorted(
        word_frequencies.items(),
        key=lambda x: x[1],
        reverse=True
    ) if freq_order == 'most' else sorted(
        word_frequencies.items(),
        key=lambda x: x[1]
    )
    top_words = sorted_word_frequencies[:top_n]

    # Generate visualizations
    unique_id = uuid.uuid4()
    wordcloud_filepath = os.path.join('static', f'wordcloud_{session["id"]}_{unique_id}.png')
    generate_wordcloud(' '.join(tokens), wordcloud_filepath)

    unigram_plot_filepath = os.path.join('static', f'unigram_plot_{session["id"]}_{unique_id}.png')
    plot_word_frequencies(top_words, f'Top {top_n} {freq_order.capitalize()} Words', unigram_plot_filepath)
    unigram_plot_url = url_for('static', filename=f'unigram_plot_{session["id"]}_{unique_id}.png')

    # Generate N-grams and their visualizations
    bigrams, trigrams = generate_ngrams(tokens)
    bigram_plot_filepath = os.path.join('static', f'bigram_plot_{session["id"]}_{unique_id}.png')
    trigram_plot_filepath = os.path.join('static', f'trigram_plot_{session["id"]}_{unique_id}.png')

    bigram_plot_url = None
    trigram_plot_url = None

    if bigrams:
        plot_ngrams(bigrams, 'Top Bigrams', bigram_plot_filepath)
        bigram_plot_url = url_for('static', filename=f'bigram_plot_{session["id"]}_{unique_id}.png')
    else:
        print("No bigrams found with enough frequency to plot.")

    if trigrams:
        plot_ngrams(trigrams, 'Top Trigrams', trigram_plot_filepath)
        trigram_plot_url = url_for('static', filename=f'trigram_plot_{session["id"]}_{unique_id}.png')
    else:
        print("No trigrams found with enough frequency to plot.")

    # Perform NER if selected
    ner_results = []
    if apply_ner:
        ner_results = extract_ner(cleaned_text, model_ner, tokenizer_ner)

    # Perform keyword extraction if selected
    keyword_results = []
    if apply_key_extraction:
        keyword_results = extract_keywords(cleaned_text, kw_model)

    # Save results in the database
    try:
        cursor = mysql.connection.cursor()

        query = """
            INSERT INTO tbl_results (
                user_id, filename, original_text, cleaned_text, word_frequencies, 
                wordcloud_path, unigram_plot_path, bigram_plot_path, trigram_plot_path, 
                created_at, processing_type, ner_results, keyword_results
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s)
        """

        word_frequencies_str = ','.join([f"{k}:{v}" for k, v in word_frequencies.items()])
        ner_results_json = json.dumps(ner_results)
        keyword_results_json = json.dumps(keyword_results)

        cursor.execute(query, (
            session['id'],
            f"{unique_id}_{uploaded_files[0].split('/')[-1]}",  # Unique filename
            merged_text,
            cleaned_text,
            word_frequencies_str,
            wordcloud_filepath,
            unigram_plot_filepath,
            bigram_plot_filepath,
            trigram_plot_filepath,
            processing_option,
            ner_results_json,
            keyword_results_json
        ))

        mysql.connection.commit()
    except Exception as e:
        print(f"Error saving results to database: {e}")
        return render_template('processing.html', message="حدث خطأ أثناء حفظ النتائج.")
    finally:
        cursor.close()

    # Finalize and return the result once processing is complete
    return render_template(
        'result.html',
        original_text=merged_text,
        cleaned_text=cleaned_text,
        wordcloud_url=url_for('static', filename=f'wordcloud_{session["id"]}_{unique_id}.png'),
        unigram_plot_url=unigram_plot_url,
        bigram_plot_url=bigram_plot_url,
        trigram_plot_url=trigram_plot_url,
        word_frequencies=top_words,
        ner_results=ner_results if ner_results else ["لا يوجد"],
        keyword_results=keyword_results if keyword_results else ["لا يوجد"]
    )


# Function to generate a unique default name like "ملف 1", "ملف 2", etc.
def generate_unique_name(user_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT default_name FROM tbl_results WHERE user_id = %s', (user_id,))
    existing_names = {row['default_name'] for row in cursor.fetchall()}

    index = 1
    while f"ملف {index}" in existing_names:
        index += 1
    return f"ملف {index}"


from flask import Response, stream_with_context


@app.route('/update_file_name/<int:result_id>', methods=['POST'])
def update_file_name(result_id):
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    data = request.get_json()
    new_name = data.get('new_name')

    if new_name:
        try:
            cursor = mysql.connection.cursor()
            cursor.execute('UPDATE tbl_results SET default_name = %s WHERE id = %s AND user_id = %s',
                           (new_name, result_id, session['id']))
            mysql.connection.commit()
            return 'Success', 200
        except Exception as e:
            print(f"Error updating the file name: {e}")
            return 'Failed to update', 500
    return 'Invalid request', 400


@app.route('/view_result/<int:result_id>', methods=['GET'])
def view_result(result_id):
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""
        SELECT original_text, cleaned_text, word_frequencies, wordcloud_path, 
               unigram_plot_path, bigram_plot_path, trigram_plot_path, 
               ner_results, keyword_results, filename
        FROM tbl_results
        WHERE id = %s AND user_id = %s
    """, (result_id, session['id']))

    result = cursor.fetchone()
    if result:
        result['word_frequencies'] = [
            tuple(item.split(':')) for item in result['word_frequencies'].split(',')
        ] if result['word_frequencies'] else []

        result['ner_results'] = json.loads(result['ner_results']) if result['ner_results'] else ["لا يوجد"]
        result['keyword_results'] = json.loads(result['keyword_results']) if result['keyword_results'] else ["لا يوجد"]
        result['filename_display'] = result['filename'].split('_', 1)[-1]  # Strip UUID from filename

        return render_template('view_result.html', result=result)
    else:
        return "Result not found."


# Full Arabic text cleaning function
def clean_arabic_text(text):
    text = mapper.map_string(text)  # Normalize using CamelTools CharMapper
    text = re.sub(r'ـ+', '', text)  # Remove Tatweel (ـــــ)
    text = re.sub(r'[ؗ-ًؚ-ْ]', '', text)  # Remove diacritics
    text = text.replace("ﻻ", "لا")  # Standardize ligatures
    text = re.sub(r'[أإآ]', 'ا', text)  # Normalize Hamzated Alif (أ, إ, آ) to bare Alif (ا)
    text = re.sub(r'ى', 'ي', text)  # Normalize Alif Maqsura (ى) to Ya (ي)
    text = re.sub(r'[^؀-ۿ\s]', '', text)  # Remove non-Arabic characters
    # Remove excess repetitions unless it's a valid double-letter word
    text = re.sub(r'(.)\1+', lambda m: m.group(0) if m.group(0) in valid_double_letter_words else m.group(1), text)
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML or special markup
    return text


# Tokenize text using Camel Tools
def tokenize_text(text):
    return simple_word_tokenize(text)


# Remove stopwords using NLTK's Arabic stopword list
def remove_stopwords(tokens):
    return [word for word in tokens if word not in arabic_stopwords]


# Perform stemming using NLTK's ISRIStemmer
def stem_words(tokens):
    return [stemmer.stem(word) for word in tokens]


# Generate word frequencies and return them
def generate_word_frequencies(tokens):
    word_counts = Counter(tokens)
    return word_counts


# Function to generate N-grams (bi-grams and tri-grams)
def generate_ngrams(tokens):
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)

    # Filter out n-grams with frequency > 1
    frequent_bigrams = [(ngram, freq) for ngram, freq in bigram_freq.items() if freq > 1]
    frequent_trigrams = [(ngram, freq) for ngram, freq in trigram_freq.items() if freq > 1]

    # Format n-grams for display
    formatted_bigrams = [(' '.join(ngram), freq) for ngram, freq in frequent_bigrams]
    formatted_trigrams = [(' '.join(ngram), freq) for ngram, freq in frequent_trigrams]

    return formatted_bigrams, formatted_trigrams


# Plot word frequencies
def plot_word_frequencies(word_frequencies, title, filepath):
    reshaped_words = [get_display(arabic_reshaper.reshape(word)) for word, _ in word_frequencies]
    freqs = [count for _, count in word_frequencies]
    plt.figure(figsize=(10, 5))
    plt.bar(reshaped_words, freqs)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, fontsize=14, ha='right')  # Align Arabic text properly
    plt.xlabel('Words', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


# Plot N-grams
def plot_ngrams(ngrams_with_freq, title, filepath, top_n=20):
    # Sort n-grams by frequency in descending order and take the top N
    sorted_ngrams = sorted(ngrams_with_freq, key=lambda x: x[1], reverse=True)[:top_n]

    if not sorted_ngrams:
        print(f"No {title.lower()} found with enough frequency to plot.")
        return
    reshaped_ngrams = [get_display(arabic_reshaper.reshape(ngram)) for ngram, _ in sorted_ngrams]
    ngram_freqs = [freq for _, freq in sorted_ngrams]
    plt.figure(figsize=(10, 5))
    plt.bar(reshaped_ngrams, ngram_freqs)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, fontsize=12, ha='right')  # Adjusted font size for readability
    plt.xlabel('N-grams', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


import os
from wordcloud import WordCloud
from bidi.algorithm import get_display
import arabic_reshaper


# Function to generate a word cloud with the correct font path
def generate_wordcloud(text, filepath):
    # Construct the full path to 'Amiri-Regular.ttf'
    font_path = os.path.join(os.path.dirname(__file__), 'Amiri-Regular.ttf')

    # Verify if the font file exists
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file not found at {font_path}")

    # Reshape and adjust text direction for Arabic
    reshaped_text = get_display(arabic_reshaper.reshape(text))

    # Generate the word cloud with the specified font
    wordcloud = WordCloud(
        font_path=font_path,
        background_color='white',
        width=800,
        height=400
    ).generate(reshaped_text)

    # Save the generated word cloud to the specified filepath
    wordcloud.to_file(filepath)


#  Named Entity Recognition (NER)
def extract_ner(text, model, tokenizer, start_token="▁"):
    tokenized_sentence = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    tokenized_sentences = tokenized_sentence['input_ids'].numpy()
    with torch.no_grad():
        output = model(**tokenized_sentence)
    last_hidden_states = output[0].numpy()
    label_indices = np.argmax(last_hidden_states[0], axis=1)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_sentences[0])
    special_tags = set(tokenizer.special_tokens_map.values())
    grouped_tokens = []
    for token, label_idx in zip(tokens, label_indices):
        if token not in special_tags:
            if not token.startswith(start_token) and len(token.replace(start_token, "").strip()) > 0:
                grouped_tokens[-1]["token"] += token
            else:
                grouped_tokens.append({"token": token, "label": custom_labels[label_idx]})
    ents = []
    prev_label = "O"
    for token in grouped_tokens:
        label = token["label"].replace("I-", "").replace("B-", "")
        if token["label"] != "O":
            if label != prev_label:
                ents.append({"token": [token["token"]], "label": label})
            else:
                ents[-1]["token"].append(token["token"])
        prev_label = label
    ents = [
        {"token": "".join(rec["token"]).replace(start_token, " ").strip(), "label": rec["label"]}
        for rec in ents
    ]
    return ents


# Keyword Extraction
def extract_keywords(text, kw_model):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=10)
    return keywords


# Profile route
# Profile route
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    user_id = session['id']

    # Fetch user details from the database
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        "SELECT first_name, second_name, username, email, id, birthdate, country, cv, major FROM tbl_users WHERE id = %s",
        (user_id,))
    user_info = cursor.fetchone()

    cursor.execute("""
        SELECT id, filename, default_name, cleaned_text, word_frequencies, wordcloud_path, 
               unigram_plot_path, bigram_plot_path, trigram_plot_path, created_at, 
               processing_type, ner_results, keyword_results
        FROM tbl_results
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (user_id,))
    user_results = cursor.fetchall()

    # Ensure default_name is set properly and format additional fields
    for result in user_results:
        if not result['default_name']:
            result['default_name'] = f"\u0645\u0644\u0641 \u062c\u062f\u064a\u062f {result['id']}"

        # Assign formatted date if available (formatted as 'day-month-year hour:minute')
        result['formatted_date'] = result['created_at'].strftime('%d-%m-%Y %H:%M') if result[
            'created_at'] else 'غير متاح'

        # Assign display filename if available (show only the original filename without the UUID prefix)
        result['display_filename'] = os.path.basename(result['filename']).split('_', 1)[-1] if result[
            'filename'] else 'غير متاح'

        # Parse word_frequencies back into a list of tuples
        result['word_frequencies'] = [
            tuple(item.split(':')) for item in result['word_frequencies'].split(',')
        ]

        # Parse NER results from JSON
        result['ner_results'] = json.loads(result['ner_results']) if result['ner_results'] else ["لا يوجد"]

        # Parse Keyword results from JSON
        result['keyword_results'] = json.loads(result['keyword_results']) if result['keyword_results'] else ["لا يوجد"]

    if request.method == 'POST':
        # Get the updated data from the form, including the major field
        birthdate = request.form.get('birthdate')
        country = request.form.get('country')
        cv = request.form.get('cv')
        major = request.form.get('major')

        try:
            # Update the user's profile information in the database
            cursor.execute("""
                UPDATE tbl_users 
                SET birthdate = %s, country = %s, cv = %s, major = %s
                WHERE id = %s
            """, (birthdate, country, cv, major, user_id))
            mysql.connection.commit()
        except Exception as e:
            print(f"Error updating profile: {e}")

    return render_template('profile.html', user=user_info, results=user_results)


@app.route('/update_biography', methods=['POST'])
def update_biography():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    user_id = session['id']
    # Get form data
    birthdate = request.form.get('birthdate', '').strip()
    country = request.form.get('country', '').strip()
    cv = request.form.get('cv', '').strip()

    # Validate the form data
    if not birthdate or not country or not cv:
        return redirect(url_for('profile', error_message="يرجى ملء جميع الحقول المطلوبة."))

    try:
        # Update the user information in the database
        cursor = mysql.connection.cursor()
        cursor.execute("""
            UPDATE tbl_users
            SET birthdate = %s, country = %s, cv = %s
            WHERE id = %s
        """, (birthdate, country, cv, user_id))
        mysql.connection.commit()
        cursor.close()

        return redirect(url_for('profile', success_message="تم تحديث السيرة الذاتية بنجاح."))
    except Exception as e:
        print(f"Error updating biography: {e}")
        return redirect(url_for('profile', error_message="حدث خطأ أثناء تحديث السيرة الذاتية."))


from flask import Flask, flash, redirect, render_template, request, session, url_for


@app.route('/update_user_info', methods=['POST'])
def update_user_info():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    user_id = session['id']
    username = request.form.get('username')
    email = request.form.get('email')
    first_name = request.form.get('first_name')
    second_name = request.form.get('second_name')
    major = request.form.get('major')  # Added the major field

    # Validate the inputs
    if not all([username, email, first_name, second_name, major]):
        return redirect(url_for('profile', error_message="يرجى ملء جميع الحقول المطلوبة."))

    try:
        cursor = mysql.connection.cursor()
        cursor.execute("""
            UPDATE tbl_users 
            SET username = %s, email = %s, first_name = %s, second_name = %s, major = %s
            WHERE id = %s
        """, (username, email, first_name, second_name, major, user_id))  # Updated to include the major field
        mysql.connection.commit()
        cursor.close()
    except Exception as e:
        print(f"Error updating user info: {e}")
        return redirect(url_for('profile', error_message="حدث خطأ أثناء تحديث المعلومات. يرجى المحاولة لاحقًا."))

    return redirect(url_for('profile', success_message="تم تحديث معلوماتك بنجاح."))


# @app.route('/profile', methods=['GET', 'POST'])
# def profile():
#     if 'loggedin' not in session:
#         return redirect(url_for('login'))
#
#     user_id = session['id']  # This should be 'id', not 'user_id'
#
#     # Fetch user details from the database
#     cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
#     cursor.execute("SELECT username, email FROM tbl_users WHERE id = %s", (user_id,))
#     user_info = cursor.fetchone()
#
#     # Fetch user results
#     cursor.execute("""
#         SELECT filename, cleaned_text, word_frequencies, wordcloud_path, unigram_plot_path, bigram_plot_path, trigram_plot_path
#         FROM tbl_results
#         WHERE user_id = %s
#         ORDER BY created_at DESC
#     """, (user_id,))
#     user_results = cursor.fetchall()
#
#     return render_template('profile.html', user=user_info, results=user_results)
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

import traceback


@app.route('/delete_file/<int:result_id>', methods=['DELETE'])
def delete_file(result_id):
    if 'loggedin' not in session:
        return jsonify({"success": False, "message": "User not logged in"}), 401

    try:
        cursor = mysql.connection.cursor()
        delete_query = 'DELETE FROM tbl_results WHERE id = %s AND user_id = %s'
        cursor.execute(delete_query, (result_id, session['id']))

        if cursor.rowcount == 0:
            print("File not found or unauthorized action")
            return jsonify({"success": False, "message": "File not found or unauthorized action"}), 404

        mysql.connection.commit()
        print("File deleted successfully")
        return jsonify({"success": True, "message": "File deleted successfully"}), 200
    except Exception as e:
        print("Error deleting file:")
        print(e)
        traceback.print_exc()  # This will give us the complete stack trace
        return jsonify({"success": False, "message": f"Error deleting file: {str(e)}"}), 500
    finally:
        cursor.close()


if __name__ == '__main__':
    app.run(debug=True)