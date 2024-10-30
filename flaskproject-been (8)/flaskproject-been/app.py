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




app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

# MySQL Configuration
# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
# Default MAMP port for MySQL
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'  # Default MAMP password (if unchanged)
app.config['MYSQL_DB'] = 'flask_users'
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






#def send_verification_email(email, verification_token):
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



# Upload and process file route
# Upload and process file route

# Function to generate a unique default name like "ملف 1", "ملف 2", etc.
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('upload.html', message='الرجاء اختيار ملف من نوع .txt')

    file = request.files['file']
    if not file.filename.endswith('.txt'):
        return render_template('upload.html', message='يرجى اختيار ملف نصي بامتداد .txt')

    processing_option = request.form.get('processing')
    top_n = int(request.form.get('top_n', 10))  # Default Top N words
    freq_order = request.form.get('freq_order', 'most')

    if file and file.filename.endswith('.txt'):
        filename = f"{uuid.uuid4()}_{file.filename}"  # Unique filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Create a unique default name using the original file's name without extension
        default_name = generate_unique_name(session['id'])

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        cleaned_text = clean_arabic_text(text)
        tokens = tokenize_text(cleaned_text)

        # Apply text processing options
        if processing_option == "clean_preprocess":
            tokens = remove_stopwords(tokens)
        elif processing_option == "clean_stem":
            tokens = remove_stopwords(tokens)
            tokens = stem_words(tokens)

        word_frequencies = generate_word_frequencies(tokens)
        sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True) if freq_order == 'most' else sorted(word_frequencies.items(), key=lambda x: x[1])
        top_words = sorted_word_frequencies[:top_n]

        # Generate a unique identifier for each upload to avoid overwriting visualizations
        unique_id = uuid.uuid4()

        # Generate visualizations and save file paths
        wordcloud_filepath = os.path.join('static', f'wordcloud_{session["id"]}_{unique_id}.png')
        generate_wordcloud(' '.join(tokens), wordcloud_filepath)

        unigram_plot_filepath = os.path.join('static', f'word_freq_plot_{session["id"]}_{unique_id}.png')
        plot_word_frequencies(top_words, f'Top {top_n} {freq_order.capitalize()} Words', unigram_plot_filepath)
        unigram_plot_url = url_for('static', filename=f'word_freq_plot_{session["id"]}_{unique_id}.png')

        # Initialize bigram and trigram plot file paths as empty strings
        bigram_plot_filepath = ''
        trigram_plot_filepath = ''

        # Generate bigrams and trigrams
        bigrams, trigrams = generate_ngrams(tokens)

        # Set bigram_plot_url based on whether bigrams exist
        if bigrams:
            bigram_plot_filepath = os.path.join('static', f'bigram_plot_{session["id"]}_{unique_id}.png')
            plot_ngrams(bigrams, 'Top Bigrams', bigram_plot_filepath)
            bigram_plot_url = url_for('static', filename=f'bigram_plot_{session["id"]}_{unique_id}.png')
        else:
            bigram_plot_url = None  # Set to None if no bigrams

        # Set trigram_plot_url based on whether trigrams exist
        if trigrams:
            trigram_plot_filepath = os.path.join('static', f'trigram_plot_{session["id"]}_{unique_id}.png')
            plot_ngrams(trigrams, 'Top Trigrams', trigram_plot_filepath)
            trigram_plot_url = url_for('static', filename=f'trigram_plot_{session["id"]}_{unique_id}.png')
        else:
            trigram_plot_url = None  # Set to None if no trigrams

        # Convert word frequencies to string format for storing in database
        word_frequencies_str = ','.join([f"{word}:{freq}" for word, freq in sorted_word_frequencies])

        # Store results in MySQL for the logged-in user, including the default name
        try:
            cursor = mysql.connection.cursor()
            cursor.execute(
                'INSERT INTO tbl_results (user_id, filename, default_name, cleaned_text, word_frequencies, wordcloud_path, unigram_plot_path, bigram_plot_path, trigram_plot_path) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',
                (session['id'], filename, default_name, cleaned_text, word_frequencies_str, wordcloud_filepath, unigram_plot_filepath, bigram_plot_filepath, trigram_plot_filepath))
            mysql.connection.commit()
        except Exception as e:
            print(f"Error saving results to database: {e}")
            return "Error saving results."

        # Render the result page
        return render_template('result.html',
                               original_text=text,
                               cleaned_text=cleaned_text,
                               wordcloud_url=url_for('static', filename=f'wordcloud_{session["id"]}_{unique_id}.png'),
                               unigram_plot_url=unigram_plot_url,
                               bigram_plot_url=bigram_plot_url,
                               trigram_plot_url=trigram_plot_url,
                               word_frequencies=top_words)

    return redirect(url_for('home'))

# Function to generate a unique default name like "ملف 1", "ملف 2", etc.
def generate_unique_name(user_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT default_name FROM tbl_results WHERE user_id = %s', (user_id,))
    existing_names = {row['default_name'] for row in cursor.fetchall()}

    index = 1
    while f"ملف {index}" in existing_names:
        index += 1
    return f"ملف {index}"

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






# Route to display individual result details
@app.route('/result/<int:result_id>', methods=['GET'])
def view_result(result_id):
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""
        SELECT filename, cleaned_text, word_frequencies, wordcloud_path, unigram_plot_path, bigram_plot_path, trigram_plot_path
        FROM tbl_results
        WHERE id = %s AND user_id = %s
    """, (result_id, session['id']))
    result = cursor.fetchone()

    if result:
        # Parse word_frequencies back into a list of tuples
        result['word_frequencies'] = [
            tuple(item.split(':')) for item in result['word_frequencies'].split(',')
        ]
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



# Generate word cloud
def generate_wordcloud(text, filepath):
    reshaped_text = get_display(arabic_reshaper.reshape(text))  # Reshape and adjust text direction for Arabic
    wordcloud = WordCloud(font_path='Amiri-Regular.ttf',
                          background_color='white',
                          width=800,
                          height=400).generate(reshaped_text)
    wordcloud.to_file(filepath)

  # Profile route
# Profile route
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    user_id = session['id']

    # Fetch user details from the database
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("SELECT first_name, second_name, username, email, id FROM tbl_users WHERE id = %s", (user_id,))
    user_info = cursor.fetchone()

    # Fetch user results (include the id in the SELECT query)
    cursor.execute("""
          SELECT id, filename, default_name, cleaned_text, word_frequencies, wordcloud_path, unigram_plot_path, bigram_plot_path, trigram_plot_path
          FROM tbl_results
          WHERE user_id = %s
          ORDER BY created_at DESC
      """, (user_id,))
    user_results = cursor.fetchall()

    # Ensure default_name is set properly and includes the ID
    for result in user_results:
        if not result['default_name']:
            result['default_name'] = f"ملف جديد {result['id']}"
        # Parse word_frequencies to a list of tuples (word, frequency)
        result['word_frequencies'] = [
            tuple(item.split(':')) for item in result['word_frequencies'].split(',')
        ]

    return render_template('profile.html', user=user_info, results=user_results)

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




# MySQL Table Definitions
# CREATE TABLE tbl_users (
#     id int(11) NOT NULL AUTO_INCREMENT,
#     username varchar(50) NOT NULL,
#     email varchar(100) NOT NULL,
#     password varchar(255) NOT NULL,
#     PRIMARY KEY (id)
# );
#
# CREATE TABLE tbl_results (
#     id int(11) NOT NULL AUTO_INCREMENT,
#     user_id int(11) NOT NULL,
#     filename varchar(255) NOT NULL,
#     wordcloud_path varchar(255) NOT NULL,
#     unigram_plot_path varchar(255) NOT NULL,
#     bigram_plot_path varchar(255) NOT NULL,
#     trigram_plot_path varchar(255) NOT NULL,
#     PRIMARY KEY (id),
#     FOREIGN KEY (user_id) REFERENCES tbl_users(id)
# );

if __name__ == '__main__':
    app.run(debug=True)