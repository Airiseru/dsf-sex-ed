__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
from bs4 import BeautifulSoup
from io import BytesIO
from gtts import gTTS
import base64
import  os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from sklearn.feature_extraction.text import CountVectorizer
import altair as alt
from bs4 import BeautifulSoup
from unidecode import unidecode
import re

nltk.download('stopwords')

#####     CONSTANTS     #####
OPENAI_APIKEY = st.secrets["OPENAI_API_KEY"]
COLOR_BLUE = '#0038a8'
COLOR_RED = '#ce1126'
# COLOR_RED = '#FFA69E'
COLOR_GRAY = '#f8f8f8'
COLOR_BLACK = "#000000"

TAGALOG_STOP_WORDS = set("applause nga ug eh yun yan yung kasi ko akin aking ako alin am amin aming ang ano anumang apat at atin ating ay bababa bago bakit bawat bilang dahil dalawa dapat din dito doon gagawin gayunman ginagawa ginawa ginawang gumawa gusto habang hanggang hindi huwag iba ibaba ibabaw ibig ikaw ilagay ilalim ilan inyong isa isang itaas ito iyo iyon iyong ka kahit kailangan kailanman kami kanila kanilang kanino kanya kanyang kapag karamihan katiyakan katulad kaya kaysa ko kong kulang kumuha kung laban lahat lamang likod lima maaari maaaring maging mahusay makita marami marapat masyado may mayroon mga minsan mismo mula muli na nabanggit naging nagkaroon nais nakita namin napaka narito nasaan ng ngayon ni nila nilang nito niya niyang noon o pa paano pababa pagitan pagkakaroon pagkatapos palabas pamamagitan panahon pangalawa para paraan pareho pataas pero pumunta pumupunta sa saan sabi sabihin sarili sila sino siya tatlo tayo tulad tungkol una walang ba eh kasi lang mo naman opo po si talaga yung wala akong di kayong wag na rin din raw daw sana nag mag na medyo medjo baka Hindi pag nya sya nyo nung parang".split())
custom_stopwords = {'ive', 'im', 'feel', 'like', 'edit', 'hahaha', 'Edit'}
stop_words = set(stopwords.words('english'))
stop_words.update(custom_stopwords)
stop_words.update(TAGALOG_STOP_WORDS)

if "lang" not in st.session_state:
    st.session_state['lang'] = "English"

if "speak" not in st.session_state:
    st.session_state['speak'] = False

if "sidebar_state" not in st.session_state:
    st.session_state['sidebar_state'] = 'collapsed'

if "age_group" not in st.session_state:
    st.session_state['age_group'] = ""

if "layout" not in st.session_state:
    st.session_state['layout'] = "centered"

lang = "English"
speak = False
volume = 0.5

mp3_fp = BytesIO()
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Loading Dataset
rappler_df = pd.read_csv("data/rappler_final.csv", converters={"topics": pd.eval, "advocs": pd.eval, 'advoc_scores': pd.eval})

reddit_df = pd.read_csv("data/reddit-posts-tokenized.csv", converters={"spacy_tokens_no_stopwords": pd.eval, 'spacy_removed_filters_tokens': pd.eval})
comments_df = pd.read_csv("data/reddit-comments-tokenized.csv", converters={"spacy_tokens_no_stopwords": pd.eval, 'spacy_removed_filters_tokens': pd.eval})

actionable_steps_df = pd.read_csv("data/actionable-steps.csv")
advocacy_description_df = pd.read_csv("data/advocacy-description.csv")
advocacy_resources_df = pd.read_csv("data/advocacy-resources.csv")

# Getting the text
reddit_all_text = reddit_all_text = [char.strip() for sub in reddit_df['spacy_tokens_no_stopwords'] for char in sub]
filtered_reddit_all_text = [char for sub in reddit_df['spacy_removed_filters_tokens'] for char in sub]
comments_all_text = [char.strip() for sub in comments_df['spacy_tokens_no_stopwords'] for char in sub]
filtered_comments_all_text = [char for sub in comments_df['spacy_removed_filters_tokens'] for char in sub]

# Dictionary for age conversion
age_convert = {
    "Children": "5 to 9 year old",
    "Preteens + Young Teens": "10 to 15 year old",
    "Older Teens": "16 to 19 year old",
    "Young Adults": "20 to 24 year old",
    "Adults": "25 year old or above"
}

#####     FUNCTIONS     #####
def get_openai_client():
    """Function to create OpenAI Client"""
    openai.api_key = OPENAI_APIKEY
    client = OpenAI(api_key=OPENAI_APIKEY)
    return client

def init_chroma_db(collection_name, db_path='sexed_data'):
    """Function to intialize the database"""
    # Create a Chroma Client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Create an embedding function
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_APIKEY, model_name="text-embedding-3-small")

    # Create a collection
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    return collection

def process_semantic_search(Q, k=3, collection=None):
    """Function to query a subset of the collection (based on a metadata)"""
    results = collection.query(
        query_texts=[Q],    # Chroma will embed this for you
        n_results=k,        # How many results to return,
        # where={f"{metadata_key}": f"{meta_val}"} # specific data only
    )
    return results

def generate_response(task, prompt, llm):
    """Function to generate a response from the LLM given a specific task and user prompt"""
    response = llm.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': f"Perform the specified task: {task}"},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response.choices[0].message.content

def generate_response_to_question(Q, text, llm, age_group):
    """Generalized function to answer a question"""
    prompt = f"""
    Provide the answer on {Q} to a {age_group} based on this context:\n\n{text}.
    You should only respond based on the given context and don't respond if you don't know the answer. Only answer what is directly being asked.
    """
    response = generate_response(Q, prompt, llm)
    return response

def ask_query(Q, llm, k=7, collection=None, age_group=None):
    """Function to go from question to query to proper answer"""
    # Get related documents
    query_result = process_semantic_search(Q=Q, k=k, collection=collection)

    # Get the text of the documents
    text = query_result['documents'][0][0]

    # Pass into GPT to get a better formatted response to the question
    response = generate_response_to_question(Q, text, llm=llm, age_group=age_group)
    # return Markdown(response)
    return response

def generate_summary(doc, llm, word_limit=None):
    """Function to ask the LLM to create a summary"""
    task = "Text Summarization"
    prompt = "Summarize this document"
    if word_limit:
        prompt += f" in {word_limit} words"

    prompt += f":\n\n{doc}"
    response = generate_response(task, prompt, llm)
    return response

def generate_short_summary(doc, llm):
    """Function to ask the LLM to create a 2-sentence summary"""
    task = "Text Summarization"
    prompt = "Summarize this document in 2 sentences:\n\n" + doc
    response = generate_response(task, prompt, llm)
    return response

def generate_translation(doc, llm, from_lang = "English", to_lang="Tagalog"):
    """Function to translate a document from English to Language of Choice"""
    task = 'Text Translation'
    prompt = f"""Translate this document from {from_lang} to {to_lang}:\n\n{doc}
    Just give the direct translation and don't add anything else to the response.
    """
    response = generate_response(task, prompt, llm)

    return response

def text_to_speech(text, lang='en'):
    """Function to make the app "speak" based on the text"""
    tts = gTTS(text, lang=lang)
    tts.write_to_fp(mp3_fp)
    sound = mp3_fp
    sound.seek(0)
    
    # Convert speech to base64 encoding
    b64 = base64.b64encode(sound.read()).decode('utf-8')

    md = f"""
        <audio id="audioTag" controls autoplay>
        <source src="data:audio/mp3;base64,{b64}"  type="audio/mpeg" format="audio/mpeg">
        </audio>
        """
    
    st.markdown(
        md,
        unsafe_allow_html=True,
    )


# Generating a word cloud
def generate_wordcloud(text, stop_words=stopwords.words('english'), freq=False, title=None):
    if freq:
        wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stop_words,
                          min_font_size=10).generate_from_frequencies(text)
    else:
        wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stop_words,
                          min_font_size=10).generate(text)
        
    plt.figure(figsize=(10, 12), facecolor=None)
    plt.imshow(wordcloud)

    if title is not None:
        plt.title(title, fontsize=20)

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# Generate n-grams frequency plot
def plot_ngrams(text, n, top_k, title, custom_stopwords=None):
    # Convert custom_stopwords to a list if it's a set
    if isinstance(custom_stopwords, set):
        custom_stopwords = list(custom_stopwords)

    # Ensure `text` is a single string
    if isinstance(text, list):
        text = ' '.join(text)  # Join the list of strings into a single string

    # Initialize the CountVectorizer with n-gram range and stop words
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words=custom_stopwords)
    X = vectorizer.fit_transform([text])
    ngrams = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1

    # Create a DataFrame of n-grams and their counts
    ngrams_df = pd.DataFrame({'ngram': ngrams, 'count': counts})
    top_ngrams_df = ngrams_df.sort_values(by='count', ascending=False).head(top_k)

    # Plotting using Altair
    chart = alt.Chart(top_ngrams_df).mark_bar().encode(
        x=alt.X('count:Q', title='Count'),
        y=alt.Y('ngram:N', sort='-x', title=None),
        color=alt.value('skyblue')
    ).properties(
        title=title
    )

    # Render plot in Streamlit
    st.altair_chart(chart, use_container_width=True)

    return top_ngrams_df['ngram'].tolist()


def generate_commentary_response(keywords_str, llm):
    """Generate a commentary response from the LLM with a word limit."""
    task = "Commentary Generation"
    prompt = (
        "As a sex education advocate, connect the following keywords to a unifying theme "
        "and provide a thoughtful commentary in no more than 4 sentences. Discuss their "
        "significance and how they impact discussions on sexual health:\n\n" + keywords_str
    )

    response = generate_response(task, prompt, llm)
    return response


def generate_age_specific_advice(commentary, age_group, llm):
    """Generate age-specific advice based on the provided keywords and age group."""
    # Define the task and prompt for the LLM
    task = "Age-Specific Advice Generation"
    
    # Customize the prompt to include the age group
    prompt = (
        f"As a sex education advocate, provide specific recommendations or steps that "
        f"this age group {age_group} can take to address the themes and issues discussed in the {commentary}. "
        f"Be concise and practical in your suggestions. Maximum of 4 sentences only."
    )
    
    # Call the LLM to generate the response
    response = generate_response(task, prompt, llm)
    return response

def return_related_articles(topics, keyword):
    for idx, topic in enumerate(topics):
        if topic == keyword:
            return idx
    return -1

def get_related_score(scores, idx):
    return scores[idx]

def get_relevant_articles(df, keyword):
    df_copy = df.copy()
    df_copy['is_keyword'] = df_copy['advocs'].apply(lambda x: return_related_articles(x, keyword))

    filtered = df_copy[df_copy['is_keyword'] != -1]

    try:
        # get the relevant score
        filtered['keyword_score'] = filtered.apply(lambda x: get_related_score(x['advoc_scores'], x['is_keyword']), axis=1)

        # sort by the relevant advocacy score
        return filtered.sort_values(by=['keyword_score'], ascending=False).reset_index().drop(columns=['index'], axis=1)
    except:
        return None
    
def preprocess_text(text):
    if not isinstance(text, str):
        return text

    text = unidecode(text)

    pattern = re.compile(r'[\t\n]|<.*?>|!function.*;|\.igframe.*}')
    text = pattern.sub(' ', text)
    text = (
        text
        .replace('&#8217;', "'")
        .replace('&#8220;', '"')
        .replace('&#8221;', '"')
        .replace('&#8216;', "'")
    )

    soup = BeautifulSoup(text, 'html.parser')
    cleaned_text = soup.get_text()
    return cleaned_text

rappler_df['title.rendered'] = rappler_df['title.rendered'].apply(preprocess_text)

##### STYLED COMPONENTS #####
html_styles = f"""
    <style>
        h1 {{
            color: {COLOR_BLUE};
            text-align: center;
            margin-bottom: 1rem;
        }}

        .advocacy-title {{
            margin-bottom: 0;
        }}

        h3 {{
            color: {COLOR_BLACK};
        }}

        .st-emotion-cache-1whx7iy p {{
            font-size: 1.125rem;
        }}

        p {{
            font-size: 1.125rem;
            text-align: justify;
        }}

        .st-emotion-cache-1v0mbdj img {{
            position: relative;
            border-radius: 50%;
        }}

        .st-emotion-cache-1v0mbdj {{
            border-radius: 50%;
            border: 5px solid transparent;
            background-image: linear-gradient(white, white), 
                            linear-gradient(to right, {COLOR_RED}, {COLOR_BLUE});
            background-origin: border-box;
            background-clip: content-box, border-box;
        }}

        .bolded {{
            font-weight: 900;
        }}

        .italicized {{
            font-style: italic;
        }}

        .tabbed {{
            margin-left: 1.75rem;
            margin-top: 0;
        }}

        .highlight {{
            background-color: #ffeb3b;
            padding: 0 5px;
            border-radius: 5px;
        }}

        .header-image {{
            width: 100%;
            height: auto;
            border-radius: 15px;
        }}

        .border-box {{
            border: 2px dashed {COLOR_BLUE};
            padding: 20px;
            margin: 20px 0;
        }}

        .column {{
            float: left;
            width: 50%;
            padding: 10px;
        }}

        .row:after {{
            content: "";
            display: table;
            clear: both;
        }}

        .color-text-dark-blue {{
            color: #0038a8;
        }}

        .st-emotion-cache-1gv3huu {{
            background-color: #133b57;
            color: white;
        }}

        .st-emotion-cache-n8e7in header, .st-emotion-cache-n8e7in li div a span, .eczjsme13, .st-emotion-cache-1whx7iy .e1nzilvr4 p {{
            color: white;
        }}

        .st-emotion-cache-1h9usn1 {{
            border-color: white;
        }}

        .st-emotion-cache-94ux81 .e16edly10 {{
            color: white;
        }}

        .app-title {{
            margin-bottom: 0;
        }}

        .tagline {{
            text-align: center;
            font-style: italic;
            color: {COLOR_RED};
        }}

        .color-red {{
            color: {COLOR_RED};
        }}

        .quote {{
            font-style: italic;
            color: grey;
        }}

        .st-emotion-cache-p5msec {{
            align-items: center;
        }}
    </style>
"""

homepage_html = f"""
<h3>About the Project</h3>
<p>Welcome to <span class='highlight'>SHELDON!</span> Our app is designed to provide accurate, accessible, and comprehensive sexual health education for everyone. Our primary goal is to support you by answering questions, offering resources, and guiding them through various aspects of sexual health in a non-judgmental manner.</p>
<br>
<p>Explore the latest news articles on sexual health issues, get involved with our advocacy efforts, and interact with our Sex Educator Chatbot for information and guidance. Whoever you are, our user-friendly app offers the resources and support you need to make informed decisions about sexual health and education.</p>
<br>

<h3>Page Descriptions</h3>
<div class="border-box">
    <div class="row">
        <div class="column">
            <h4 class='color-red'>Advocacy Page</h4>
            <p>Contains the latest news articles about sexual health issues along with highlights of our advocacy for better sexual health and education. Stay informed and engaged with our curated content and advocacy initiatives.</p>
        </div>
        <div class="column">
            <h4 class='color-red'>Sex Education Chatbot</h4>
            <p>Provides personalized responses and educational content on sexual health topics, tailored to different age groups and individual needs. The chatbot offers reliable information and answers to your questions, ensuring that you have access to accurate and relevant sexual health education.</p>
        </div>
    </div>
</div>
"""

hide_bar = """
           <style>
           [data-testid='stSidebar'] {visibility:hidden;}
           [data-testid='collapsedControl'] {visibility:hidden;}
           </style>
           """

show_bar = """
           <style>
           [data-testid='stSidebar'] {visibility:visible;}
           [data-testid='collapsedControl'] {visibility:visible;}
           [data-testid='stSidebarNavItems'] header {color: white;}
           [data-testid='stSidebarNavLink'] span {color: white;}
           [data-testid='stVerticalBlock'] .st-emotion-cache-1whx7iy p {color: white;}
           [data-testid='stSidebarUserContent'] .st-emotion-cache-ue6h4q .st-emotion-cache-1whx7iy p {color: white;}
           [data-testid='stSidebarUserContent'] .st-emotion-cache-y4bq5x .st-emotion-cache-187vdiz p {color: white;}
           [data-testid='stWidgetLabel'] .st-emotion-cache-187vdiz p {color:black;}
           [data-testid='stTabs'] .st-emotion-cache-1whx7iy p {color: black;}
           [data-testid='stSelectbox'] .st-emotion-cache-1whx7iy p {color: black;}
           [data-testid='stExpanderToggleIcon'] .st-emotion-cache-1pbsqtx {margin: auto;}
           </style>
           """

advocacy_action_title = f"""
<style>
    .action-title {{
        text-align:center;
    }}
</style>
<h3 class='action-title'>It is time for you to take action now!</h3>
"""

advocacy_learn_more_title = "<h3 class='action-title'>Learn more</h3>"

advocacy_related_articles_title = "<h3 class='action-title'>Read related articles</h3>"

# Reminder modal dialog
@st.experimental_dialog("❗ Reminder!", width='large')
def go_main_page(age_grp):
    st.write("While our app provides information about sexual health, it is important to do your own research and seek professional medical advice if needed. Always consult a medical professional before jumping into any conclusion.")

    st.write("Always look at different resources and fact check to ensure that all the information you believe in are true.")

    st.session_state['age_group'] = age_convert[age_grp]
    
    acknowledged = st.checkbox("I understand to do my own research and not just rely on the information mentioned by the app.")
    enter = st.button("Visit SHELDON!", type='primary')

    # Button was pressed and checkbox was ticked
    if (enter and acknowledged):
        st.session_state['sidebar_state'] = 'expanded'
        st.session_state['layout'] = "wide"
        st.rerun()

    # Button was pressed but checkbox was not ticked
    elif (enter and not acknowledged):
        st.error('Please acknowledge to proceed.', icon="🚨")


# Homepage
def home():
    st.markdown("<h1 class='app-title'>Sexual Health Education Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h5 class='tagline'>Breaking Barriers, Building Knowledge: Sexual Health Education Made Easy!</h5>", unsafe_allow_html=True)
    st.write("---")
    st.markdown(homepage_html, unsafe_allow_html=True)

# Advocacy page
def advocacy_page():
    lang = st.session_state['lang']
    advoc_title = "Why Sex Education"
    intro_quote = "In a country where conservatives worry about morbidity and promiscuity against sex education, in which I as an author pays my utmost respect, lies the truth, that there are two million teenage Filipina girls who are pregnant at this moment while there are approximately four million Filipinos aged 15-19 have already had sexual intercourse and now having sex while you are reading this. We can’t do anything about it. We cannot stop it. The least option we could possibly do is to educate them."
    intro_data = "According to 2023 data from the Department of Health, the Philippines has seen a significant rise in HIV cases, particularly among young people aged 15-24. The United Nations Population Fund (UNFPA) also reported that the country continues to have one of the highest teenage pregnancy rates in Southeast Asia. These alarming statistics underscore the urgent need for comprehensive sex education to equip young people with the knowledge and skills necessary to make informed decisions about their sexual health (Feje, 2024)."

    if lang != "English":
        advoc_title = generate_translation(advoc_title, llm, "English", lang)
        intro_quote = generate_translation(intro_quote, llm, "English", lang)
        intro_data = generate_translation(intro_data, llm, "English", lang)

    st.markdown(f"<h1 class='advocacy-title'>{advoc_title}?</h1>", unsafe_allow_html=True)

    st.markdown(f"<p class='quote'>{intro_quote} -- <a href='https://thegreatmustache.wordpress.com/must-read-articles/why-philippines-needs-sex-education/', style='color: grey;'>jasmin</a> (2012)</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.write(intro_data)

    bigram_posts, trigram_posts = st.tabs(["Reddit Posts Bigram", "Reddit Posts Trigram"])
    
    with bigram_posts:
        filtered_posts = st.toggle("Remove Filter Words", key='filter-bigram-posts')
        if filtered_posts:
            top_bigrams = plot_ngrams(filtered_reddit_all_text, n=2, top_k=10, title="Top Bigrams for Reddit Posts (Filtered)", custom_stopwords=stop_words)
            commentary = """The unifying theme connecting these keywords is the empowerment and education necessary for informed decisions regarding sexual health and family planning. Birth control and the Yuzpe method exemplify the importance of understanding contraceptive options, while phrases like "tt time" and "know wt" emphasize timing and awareness in reproductive health. The availability and knowledge of pregnancy tests further enhance individuals' ability to take charge of their reproductive choices. In discussions about sexual health, promoting awareness and communication—encapsulated in "say tt"—is crucial for fostering a responsible approach to contraception and pregnancy among young people."""
        else:
            top_bigrams = plot_ngrams(reddit_all_text, n=2, top_k=10, title="Top Bigrams for Reddit Posts", custom_stopwords=stop_words)
            commentary = "The plot converge around the fundamental theme of informed sexual health practices and responsibility. Safe sex and birth control are imperative for preventing unplanned pregnancies and reducing the risk of sexually transmitted infections, highlighting the importance of being 'sexually active' in a mindful and informed manner. The mention of methods like the 'Yuzpe method' emphasizes the need for accessible information on contraceptives, while phrases like 'unprotected sex' and 'practice safe' serve as critical reminders of the consequences of inadequate sexual health awareness. Ultimately, being well-informed—captured in terms like 'know wt' and 'told tt'—empowers individuals to make healthier choices and fosters an environment where open discussions about sexual health can thrive, significantly impacting community wellness."
        
        if lang != "English":
            commentary = generate_translation(commentary, llm, "English", lang)

        st.write(commentary)

    with trigram_posts:
        filtered_posts = st.toggle("Remove Filter Words", key='filter-trigram-posts')
        if filtered_posts:
            top_trigrams = plot_ngrams(filtered_reddit_all_text, n=3, top_k=10, title="Top Trigrams for Reddit Posts (Filtered)", custom_stopwords=stop_words)
            commentary = 'The main theme of the plot revolves around the importance of informed sexual health practices among adolescents. The discussion surrounding safe sex and various birth control methods, including birth control pills and implants, is crucial, especially as young individuals, like a 17-year-old or their cousins, navigate their sexual lives. These conversations can empower youth to make informed choices, emphasizing that practicing safe sex is not just about preventing pregnancy but also about fostering healthy relationships and communication. Long story short, providing accessible information about barrier-free sex and alternative methods like the implant ensures that minors can make evidence-based decisions that prioritize their health and well-being.'
        else:
            top_trigrams = plot_ngrams(reddit_all_text, n=3, top_k=10, title="Top Trigrams for Reddit Posts", custom_stopwords=stop_words)
            commentary = 'The unifying theme connecting these keywords is the importance of accessible sexual health education tailored to adolescents. For a 17-year-old, understanding various birth control methods, including barrier methods and long-acting options like the implant, is crucial for making informed choices about safe sex. It is essential to foster open discussions about these topics, especially for teenagers who may look up to older cousins for guidance. Ultimately, promoting comprehensive education on these issues can empower young individuals to practice safe sex confidently and responsibly, reducing risks of STIs and unplanned pregnancies.'

        if lang != "English":
            commentary = generate_translation(commentary, llm, "English", lang)

        st.write(commentary)
    
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(advocacy_action_title, unsafe_allow_html=True)
    age_group = st.session_state['age_group']
    filtered_steps = actionable_steps_df[(actionable_steps_df['age_group'] == age_group) & (actionable_steps_df['language'] == lang)]
    actionable_steps = filtered_steps['actionable_steps'].iloc[0]
    st.write(actionable_steps)


    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(advocacy_learn_more_title, unsafe_allow_html=True)
    intro3 = "Choose an advocacy to learn more about"
    
    if lang != "English":
        intro3 = generate_translation(intro3, llm, from_lang="English", to_lang=lang)

    chosen_advocacy = st.selectbox(        
        intro3,
        ['Access to Contraception and Reproductive Health Services','Comprehensive Sex Education',
        'Consent and Healthy Relationships', 'Sexual Orientation and Gender Identity',
        'Sexual Health and Rights', 'Sexual Violence Prevention and Support',
        'Parental and Caregiver Involvement', 'Equity and Access', 'Others']
    )

    filtered_df = None
    if chosen_advocacy:
        filtered_df = get_relevant_articles(rappler_df, chosen_advocacy)

        resources = advocacy_resources_df[(advocacy_resources_df['age_group'] == age_group) & (advocacy_resources_df['advocacy'] == chosen_advocacy)].iloc[0]['resources']

        if lang != "English":
            resources = generate_translation(resources, llm, "English", lang)

    # Detailed description of the selected advocacy
    lang = st.session_state['lang']
    filtered_description = advocacy_description_df[(advocacy_description_df['chosen_advocacy'] == chosen_advocacy) & (advocacy_description_df['language'] == lang)]
    advocacy_description = filtered_description['description'].iloc[0]
    st.write(advocacy_description)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.popover("Some recommended resources", use_container_width=True):
        st.markdown(resources)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(advocacy_related_articles_title, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if filtered_df is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            # Article 1
            first_title = filtered_df.iloc[0]['title.rendered']
            first_date = f"Date Published: {filtered_df.iloc[0]['date']}"
            first_content = filtered_df.iloc[0]['content.rendered']
            first_link = filtered_df.iloc[0]['link']  # Assuming there's a 'link' field
            
            # Generate a short summary
            first_summary = generate_short_summary(first_content, llm)
            
            # Use the title for the popover label
            with st.popover(first_title, use_container_width=True):
                # Display title and date with formatting
                st.markdown(f"<h3>{first_title}</h3><p style='font-size:16px; color:gray;'>{first_date}</p>", unsafe_allow_html=True)
                st.write(first_summary)
                st.markdown(f"[Read more]({first_link})", unsafe_allow_html=True)
        
            # Article 4
            fourth_title = filtered_df.iloc[3]['title.rendered']
            fourth_date = f"Date Published: {filtered_df.iloc[3]['date']}"
            fourth_content = filtered_df.iloc[3]['content.rendered']
            fourth_link = filtered_df.iloc[3]['link']
            
            # Generate a short summary
            fourth_summary = generate_short_summary(fourth_content, llm)
            
            with st.popover(fourth_title, use_container_width=True):
                st.markdown(f"<h3>{fourth_title}</h3><p style='font-size:16px; color:gray;'>{fourth_date}</p>", unsafe_allow_html=True)
                st.write(fourth_summary)
                st.markdown(f"[Read more]({fourth_link})", unsafe_allow_html=True)
        
        with col2:
            # Article 2
            second_title = filtered_df.iloc[1]['title.rendered']
            second_date = f"Date Published: {filtered_df.iloc[1]['date']}"
            second_content = filtered_df.iloc[1]['content.rendered']
            second_link = filtered_df.iloc[1]['link']
            
            # Generate a short summary
            second_summary = generate_short_summary(second_content, llm)
            
            with st.popover(second_title, use_container_width=True):
                st.markdown(f"<h3>{second_title}</h3><p style='font-size:16px; color:gray;'>{second_date}</p>", unsafe_allow_html=True)
                st.write(second_summary)
                st.markdown(f"[Read more]({second_link})", unsafe_allow_html=True)
        
            # Article 5
            fifth_title = filtered_df.iloc[4]['title.rendered']
            fifth_date = f"Date Published: {filtered_df.iloc[4]['date']}"
            fifth_content = filtered_df.iloc[4]['content.rendered']
            fifth_link = filtered_df.iloc[4]['link']
            
            # Generate a short summary
            fifth_summary = generate_short_summary(fifth_content, llm)
            
            with st.popover(fifth_title, use_container_width=True):
                st.markdown(f"<h3>{fifth_title}</h3><p style='font-size:16px; color:gray;'>{fifth_date}</p>", unsafe_allow_html=True)
                st.write(fifth_summary)
                st.markdown(f"[Read more]({fifth_link})", unsafe_allow_html=True)
        
        with col3:
            # Article 3
            third_title = filtered_df.iloc[2]['title.rendered']
            third_date = f"Date Published: {filtered_df.iloc[2]['date']}"
            third_content = filtered_df.iloc[2]['content.rendered']
            third_link = filtered_df.iloc[2]['link']
            
            # Generate a short summary
            third_summary = generate_short_summary(third_content, llm)
            
            with st.popover(third_title, use_container_width=True):
                st.markdown(f"<h3>{third_title}</h3><p style='font-size:16px; color:gray;'>{third_date}</p>", unsafe_allow_html=True)
                st.write(third_summary)
                st.markdown(f"[Read more]({third_link})", unsafe_allow_html=True)
        
            # Article 6
            sixth_title = filtered_df.iloc[5]['title.rendered']
            sixth_date = f"Date Published: {filtered_df.iloc[5]['date']}"
            sixth_content = filtered_df.iloc[5]['content.rendered']
            sixth_link = filtered_df.iloc[5]['link']
            
            # Generate a short summary
            sixth_summary = generate_short_summary(sixth_content, llm)
            
            with st.popover(sixth_title, use_container_width=True):
                st.markdown(f"<h3>{sixth_title}</h3><p style='font-size:16px; color:gray;'>{sixth_date}</p>", unsafe_allow_html=True)
                st.write(sixth_summary)
                st.markdown(f"[Read more]({sixth_link})", unsafe_allow_html=True)

        with st.popover("View all related articles", use_container_width=True):
            st.dataframe(filtered_df[['title.rendered', 'date', 'link']])

    else:
        st.write("No related articles...")

# Chatbot
def chatbot_page():
    # Set variables
    user_prompt = "Ask a question"
    load_response = "Loading a response..."
    lang = st.session_state['lang']
    age_group = st.session_state['age_group']

    st.title("Hi, I am SHELDON!")
    st.write("---")
    
    # Set a default model
    if 'openai_model' not in st.session_state:
        st.session_state['openai_model'] = 'gpt-3.5-turbo'
    
    # Initialize feedback
    if "feedback" not in st.session_state:
        st.session_state['feedback'] = None
    
    # Initiliaze spoken
    if "spoken" not in st.session_state:
        st.session_state['spoken'] = False
    
    # Initialize total number of responses
    if "total_responses" not in st.session_state:
        st.session_state['total_responses'] = 0

    # Initialize chat history
    if "messages" not in st.session_state:
        initial_message = "Hi! I am Sheldon and I am here to assist you with any question related to sexual health, sexual rights, HIV, and many more! What would you like to know?"

        if lang != "English":
            initial_message = generate_translation(initial_message, llm, "English", lang)

        st.session_state['messages'] = [{"role": "assistant", "content": initial_message}]

    # Display chat messages from history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message['role']):
            st.markdown(message['content'])

            if message['role'] == 'assistant':
                # Text to Speech the response (if enabled)
                if st.session_state['speak'] and (i == len(st.session_state.messages) - 1) and not(st.session_state['spoken']):
                    st.session_state['spoken'] = True
                    if st.session_state['lang'] == 'English':
                        text_to_speech(message['content'], lang='en')
                    else:
                        # tl = Filipino (the only one available)
                        text_to_speech(message['content'], lang='tl')

                # insert code for feedback page
    
    # Accept user input
    if prompt := st.chat_input(user_prompt):
        # Reset feedback
        st.session_state.feedback = None

        # Add user message to chat history
        st.session_state.messages.append({"role": "user",
                                          "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Translate the user prompt to english
        if lang != "English":
            prompt = generate_translation(prompt, llm, lang, "English")

        # Display response
        with st.chat_message("assistant"):
            with st.spinner(load_response):
                response = ask_query(prompt, llm, k=7, collection=collection, age_group=age_group)

                # Translate the response if the language of choice is not in English
                if lang != "English":
                    response = generate_translation(response, llm, "English", lang)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant",
                                                  "content": response})
        
        st.session_state['spoken'] = False
        st.rerun()


#####     MAIN SITE     #####
CHROMA_DATA_PATH = 'sexed_data'
COLLECTION_NAME = "sexed_data"

# Initialize ChromaDB client
collection = init_chroma_db(collection_name="sexed_data", db_path="sexed_data")

# Initialize OpenAI client
llm = get_openai_client()

# Create streamlit app
st.set_page_config(layout=st.session_state.layout, initial_sidebar_state=st.session_state.sidebar_state)
st.markdown(html_styles, unsafe_allow_html=True)

# Set font style of the app
with open ("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# Initial page to get the age group of user
if st.session_state['age_group'] == "":
    st.session_state['sidebar_state'] = 'collapsed'
    st.markdown(hide_bar, unsafe_allow_html=True)
    st.title("SHELDON")
    st.subheader("Welcome to SHELDON!")
    st.markdown(
        "<p>Short for the <span class='bolded' style='color: #5C7AFF;'><span class='color-text-dark-blue'>s</span>exual <span class='color-text-dark-blue'>he</span>a<span class='color-text-dark-blue'>l</span>th e<span  class='color-text-dark-blue'>d</span>ucati<span class='color-text-dark-blue'>on</span> chatbot</span>, which is an app that aims to provide accurate, accessible, and comprehensive sexual health education for everyone! Whether you are a student, parent, or simply someone seeking reliable information, our user-friendly app offers the resources and support you need to learn more about sexual health to make informed decisions and reduce the stigma around sexual topics.</p>",
        unsafe_allow_html=True

    )

    st.subheader("Getting Started")

    age_grp = st.radio(
        "To start, please select the appropriate profile:",
        ["Children", "Preteens + Young Teens", "Older Teens", "Young Adults", "Adults"],
        captions=[
            "Kids from ages 5-9 years old",
            "Teens from ages 10-15 years old",
            "Teens from ages 16-19 years old",
            "Adults from ages 20-24 years old",
            "Adults ages 25 years old and above"
        ],
        index=None
    )
    next_btn = st.button("Next")

    if (next_btn and (age_grp is not None)):
        go_main_page(age_grp)
    elif (next_btn and (age_grp is None)):
        st.error('Please select your age group to proceed.', icon="🚨")

# Rest of the app
else:
    st.markdown(show_bar, unsafe_allow_html=True)
    homepage = st.Page(home, title="Home", icon=":material/home:")
    advocacypage = st.Page(advocacy_page, title="Advocacy Page", icon=":material/psychology_alt:")
    chatpage = st.Page(chatbot_page, title="Chatbot", icon=":material/forum:")

    pg = st.navigation(
        {
            "Navigation": [homepage, advocacypage, chatpage]
        }
    )

    with st.sidebar.expander("⚙️ Response Settings"):
        lang = st.selectbox(
            "Language Options", ["English", "Tagalog", "Cebuano", "Hiligaynon", "Ilocano"]
        )
        st.session_state['lang'] = lang
        speak = st.toggle("Text to Speech", value=st.session_state['speak'])
        st.session_state['speak'] = speak

    pg.run()