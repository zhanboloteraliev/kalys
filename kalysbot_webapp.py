import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# --- Configuration ---
INDEX_NAME = "kalysbot"
DIMENSION = 1536
LLM_MODEL = "gpt-3.5-turbo"
PDF_DIR = "data/pdfs"
DEFAULT_LANGUAGE = "en"

# --- Language Localization Maps ---
LANGUAGE_MAPS = {
    "en": {
        "label": "🇬🇧 English",
        "page_title": "Kalys MVP",
        "main_title": "I am Kalys 👋, a Kyrgyz law assistant bot.",
        "caption_connected": "Connected to Pinecone index: {INDEX_NAME}. Your knowledge base is ready!",
        "sidebar_header_filter": "Document Filtering (Knowledge Base)",
        "sidebar_markdown_filter": "Select the law codes the engine should search within.",
        "sidebar_info_searching": "Searching across **{count}** documents.",
        "sidebar_header_retrieval": "Retrieval Settings",
        "sidebar_slider_label": "Documents Chunks to Retrieve (k)",
        "sidebar_header_downloads": "Source Downloads",
        "sidebar_markdown_downloads": "Download the original law documents below for verification.",
        "sidebar_warning_dir": f"Local directory '{PDF_DIR}' not found. Cannot offer downloads.",
        "sidebar_info_no_pdfs": f"No PDF files found in the '{PDF_DIR}' directory.",
        "sidebar_check_api": "Ensure your API keys are set in a local `.env` file.",
        "chat_input_placeholder": "Ask a question about laws...",
        "spinner_thinking": "Thinking...",
        "source_expander": "📝 Sources Used for this Answer",
        "source_label": "Source {j}: {display_name} (Page {display_page})",
        "button_new_chat": "Start New Chat",
        "status_success": "Successfully initialized RAG Chain with document filtering and custom prompt!",
        "error_no_doc_selected": "No documents selected for search. Please select at least one document.",
        "error_missing_keys": "Missing one or more required API keys/environment variables. Check your `.env` file.",
        "error_rag_setup": "Error connecting to Pinecone or setting up RAG: {e}",
        "error_generation": "An error occurred during generation: {e}",
        "error_chain_fail": "RAG chain failed to initialize. Please check document selection, API keys, and index status.",
        "warning_disclaimer": "⚠️ Always verify law status and accuracy with official sources.",
    },
    "ru": {
        "label": "🇷🇺 Русский",
        "page_title": "Калыс MVP",
        "main_title": "Я Калыс 👋, бот-помощник по законам Кыргызстана.",
        "caption_connected": "Подключено к индексу Pinecone: {INDEX_NAME}. Ваша база знаний готова!",
        "sidebar_header_filter": "Фильтрация Документов (База знаний)",
        "sidebar_markdown_filter": "Выберите своды законов для поиска.",
        "sidebar_info_searching": "Поиск ведется по **{count}** документам.",
        "sidebar_header_retrieval": "Настройки Извлечения",
        "sidebar_slider_label": "Извлекаемые Части Документов (k)",
        "sidebar_header_downloads": "Загрузка Источников",
        "sidebar_markdown_downloads": "Загрузите исходные юридические документы для проверки.",
        "sidebar_warning_dir": f"Локальный каталог '{PDF_DIR}' не найден. Загрузка недоступна.",
        "sidebar_info_no_pdfs": f"В каталоге '{PDF_DIR}' файлы PDF не найдены.",
        "sidebar_check_api": "Убедитесь, что ваши ключи API установлены в локальном файле `.env`.",
        "chat_input_placeholder": "Задайте вопрос про законы...",
        "spinner_thinking": "Думаю...",
        "source_expander": "📝 Источники, использованные для ответа",
        "source_label": "Источник {j}: {display_name} (Стр. {display_page})",
        "button_new_chat": "Начать Новый Чат",
        "status_success": "RAG-цепочка успешно инициализирована с фильтрацией документов и пользовательским запросом!",
        "error_no_doc_selected": "Для поиска не выбраны документы. Пожалуйста, выберите хотя бы один документ.",
        "error_missing_keys": "Отсутствует один или несколько требуемых ключей API/переменных среды. Проверьте ваш файл `.env`.",
        "error_rag_setup": "Ошибка подключения к Pinecone или настройки RAG: {e}",
        "error_generation": "Произошла ошибка при генерации: {e}",
        "error_chain_fail": "RAG-цепочка не инициализирована. Проверьте выбор документов, ключи API и статус индекса.",
        "warning_disclaimer": "⚠️ Всегда проверяйте актуальность и точность законов по официальным источникам.",
    },
    "ky": {
        "label": "🇰🇬 Кыргызча",
        "page_title": "Калыс MVP",
        "main_title": "Мен Калыс 👋 — Кыргыз мыйзамдары боюнча жардамчы бот.",
        "caption_connected": "Pinecone индексине туташты: {INDEX_NAME}. Сиздин билим базаңыз даяр!",
        "sidebar_header_filter": "Документтерди чыпкалоо (Билим базасы)",
        "sidebar_markdown_filter": "Издөө үчүн мыйзам кодекстерин тандаңыз.",
        "sidebar_info_searching": "**{count}** документ боюнча издөө.",
        "sidebar_header_retrieval": "Издөө Жөндөөлөрү",
        "sidebar_slider_label": "Канча булактарды колдонуп жооп тапкыңыз келет?",
        "sidebar_header_downloads": "Булактарды Жүктөп Алуу",
        "sidebar_markdown_downloads": "Текшерүү үчүн баштапкы юридикалык документтерди жүктөп алыңыз.",
        "sidebar_warning_dir": f"Жергиликтүү '{PDF_DIR}' каталогу табылган жок. Жүктөө мүмкүн эмес.",
        "sidebar_info_no_pdfs": f"'{PDF_DIR}' каталогунда PDF файлдары табылган жок.",
        "sidebar_check_api": "Сиздин API ачкычтарыңыз жергиликтүү `.env` файлында орнотулганын текшериңиз.",
        "chat_input_placeholder": "Мыйзамдар боюнча суроо бериңиз...",
        "spinner_thinking": "Ойлонууда...",
        "source_expander": "📝 Колдонулган булактар",
        "source_label": "Булак {j}: {display_name} (Бет {display_page})",
        "button_new_chat": "Жаңы Чатты Баштоо",
        "status_success": "RAG чынжырчасы документти чыпкалоо жана жеке суроо менен ийгиликтүү ишке киргизилди!",
        "error_no_doc_selected": "Издөө үчүн документтер тандалган жок. Сураныч, жок дегенде бир документти тандаңыз.",
        "error_missing_keys": "Талап кылынган API ачкычтары/айлана-чөйрө өзгөрмөлөрү жок. Сиздин `.env` файлыңызды текшериңиз.",
        "error_rag_setup": "Pinecone'го туташууда же RAG'ди орнотууда ката: {e}",
        "error_generation": "Генерация учурунда ката кетти: {e}",
        "error_chain_fail": "RAG чынжырчасы ишке киргизилген жок. Документ тандоону, API ачкычтарын жана индекс статусун текшериңиз.",
        "warning_disclaimer": "⚠️ Мыйзамдардын статусун жана тактыгын расмий булактардан текшериңиз.",
    },
}

# --- Set up the current language and translator function ---
if "language" not in st.session_state:
    st.session_state["language"] = DEFAULT_LANGUAGE


# Translator function
def T(key, **kwargs):
    """Returns the translated string for a given key, optionally applying formatting."""
    lang_map = LANGUAGE_MAPS.get(st.session_state.language, LANGUAGE_MAPS[DEFAULT_LANGUAGE])
    text = lang_map.get(key, LANGUAGE_MAPS[DEFAULT_LANGUAGE].get(key, f"[MISSING TEXT: {key}]"))
    return text.format(**kwargs)


# --- NEW: LLM Settings for Detailed Answers ---
LLM_TEMPERATURE = 0.5
SYSTEM_PROMPT = (
    "You are Kalys, an expert legal assistant. Your primary function is to provide comprehensive, "
    "detailed, and accurate answers based *only* on the provided source documents (laws and codes). "
    "Structure your answers clearly, similar to a detailed report or a ChatPDF response, ensuring all "
    "relevant legal context is included. Do not answer questions that cannot be supported by the sources."
)

# --- Define the Custom QA Prompt Template ---
QA_PROMPT_TEMPLATE = (
        SYSTEM_PROMPT +
        "\n\nUse the following pieces of context to answer the question at the end. "
        "If you don't know the answer, just state that you don't know, don't try to make up an answer."
        "\n\n----------------\n"
        "{context}"
        "\n----------------\n"
        "Question: {question}"
        "\nHelpful Answer:"
)

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# --- Mapping for Descriptive Names (Localized) ---
# Each file now maps to an object containing the localized name for each supported language.
PDF_NAME_MAP = {
    "01aug2020_kidscode.pdf": {"en": "Children's Code", "ru": "Кодекс о детях", "ky": "Балдар жөнүндө кодекс"},
    "06aug2025_civilprocedurecode.pdf": {"en": "Civil Procedural Code", "ru": "Гражданский процессуальный кодекс",
                                         "ky": "Жарандык процесстик кодекси"},
    "09aug2025_crimeprocedurecode.pdf": {"en": "Criminal Procedure Code", "ru": "Уголовно-процессуальный кодекс",
                                         "ky": "Кылмыш-жаза процессуалдык кодекси"},
    "10jul2025_labourcode.pdf": {"en": "Labour Code", "ru": "Трудовой кодекс", "ky": "Эмгек кодекси"},
    "10jul2025_nontaxrevenuecode.pdf": {"en": "Non-Tax Income Code", "ru": "Кодекс о неналоговых доходах",
                                        "ky": "Салыктык эмес кирешелер жөнүндө кодекс"},
    "13aug2025_offencecode.pdf": {"en": "Code of Offenses", "ru": "Кодекс о правонарушениях",
                                  "ky": "Укук бузуулар жөнүндө кодекс"},
    "17jul2025_familycode.pdf": {"en": "Family Code", "ru": "Семейный кодекс", "ky": "Үй-бүлө кодекси"},
    "28feb2025_adminprocedurecode.pdf": {"en": "Administrative Procedural Code",
                                         "ru": "Административный процессуальный кодекс",
                                         "ky": "Административдик-процесстик кодекси"},
    "28jul2025_penalcode.pdf": {"en": "Criminal Code", "ru": "Уголовный кодекс", "ky": "Кылмыш-жаза кодекси"},
    "29apr2025_budgetcode.pdf": {"en": "Budget Code", "ru": "Бюджетный кодекс", "ky": "Бюджет кодекси"},
    "31jul2025_crimeexecutivecode.pdf": {"en": "Criminal Execution Code", "ru": "Уголовно-исполнительный кодекс",
                                         "ky": "Жаза-аткаруу кодекси"},
    "31jul2025_taxcode.pdf": {"en": "Tax Code", "ru": "Налоговый кодекс", "ky": "Салык кодекси"},
    "05052021_constitution.pdf": {"en": "Constitution of Kyrgyz Republic", "ru": "Конституция Кыргызской Республики",
                                  "ky": "Кыргыз Республикасынын Конституциясы"},
    "07022025_civilcode2.pdf": {"en": "Civil Code (Part 2)", "ru": "Гражданский кодекс (часть 2)",
                                "ky": "Жарандык кодекс (2-бөлүк)"},
    "07102025_civilcode1.pdf": {"en": "Civil Code (Part 1)", "ru": "Гражданский кодекс (часть 1)",
                                "ky": "Жарандык кодекс (1-бөлүк)"},
    "digitalcode.pdf": {"en": "Digital Code", "ru": "Цифровой кодекс", "ky": "Санарип кодекси"},
    "judicalhonorcode.pdf": {"en": "Code of Ethics For Judges", "ru": "Кодекс чести судей",
                             "ky": "Соттордун ар-намыс кодекси"},
    "landcode.pdf": {"en": "Land Code", "ru": "Земельный кодекс", "ky": "Жер кодекси"},
    "tkeaes.pdf": {"en": "Customs Code of the Eurasian Economic Union",
                   "ru": "Таможенный кодекс Евразийского экономического союза",
                   "ky": "Евразия экономикалык биримдигинин Бажы кодекси"}
}


# --- New Helper Function for Localized Document Names ---
def get_doc_display_name(file_name):
    """Retrieves the localized display name for a given file name."""
    lang = st.session_state.language
    doc_map = PDF_NAME_MAP.get(file_name, {})
    # Fallback: Check current language, then English, then raw file name
    return doc_map.get(lang, doc_map.get(DEFAULT_LANGUAGE, file_name))


# --- Load Environment Variables from .env file ---
load_dotenv()

# --- Streamlit Page Configuration ---
# Use localized page title
st.set_page_config(
    page_title=T("page_title"),
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# --- UI Components ---
# Use localized main title
st.title(T("main_title"))
st.caption(T("caption_connected", INDEX_NAME=INDEX_NAME))

# --- Session State for Document Selection ---
if "selected_files" not in st.session_state:
    st.session_state["selected_files"] = list(PDF_NAME_MAP.keys())

# --- Sidebar for Configuration and Downloads ---
with st.sidebar:
    # Language Selector
    st.header("🌐 Language")

    # Get the current index of the selected language
    lang_labels = [LANGUAGE_MAPS[key]["label"] for key in LANGUAGE_MAPS.keys()]
    current_index = list(LANGUAGE_MAPS.keys()).index(st.session_state.language)

    selected_label = st.selectbox(
        "Select UI Language",
        options=lang_labels,
        index=current_index,
        label_visibility="collapsed"
    )

    # Update session state if language changes
    selected_lang_key = [key for key, value in LANGUAGE_MAPS.items() if value["label"] == selected_label][0]
    if st.session_state.language != selected_lang_key:
        st.session_state.language = selected_lang_key
        st.rerun()  # Rerun to apply new language strings immediately

    # Document Filtering
    st.header(T("sidebar_header_filter"))
    st.markdown(T("sidebar_markdown_filter"))

    selected_files_temp = []
    for file_name in PDF_NAME_MAP.keys():
        default_state = file_name in st.session_state.selected_files

        # Use helper function to get localized display name
        display_name = get_doc_display_name(file_name)

        # Checkbox labels use the descriptive map names
        if st.checkbox(f" {display_name}", value=default_state, key=f"checkbox_{file_name}"):
            selected_files_temp.append(file_name)

    st.session_state.selected_files = selected_files_temp
    st.info(T("sidebar_info_searching", count=len(st.session_state.selected_files)))

    # Retrieval Settings
    st.markdown("---")
    st.header(T("sidebar_header_retrieval"))
    k_retrieval = st.slider(T("sidebar_slider_label"), 1, 10, 4, 1)

    # Source Downloads
    st.markdown("---")
    st.header(T("sidebar_header_downloads"))
    st.markdown(T("sidebar_markdown_downloads"))

    if os.path.isdir(PDF_DIR):
        try:
            pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

            if pdf_files:
                for pdf_file_name in pdf_files:
                    file_path = os.path.join(PDF_DIR, pdf_file_name)

                    # Use helper function to get localized display name for download button
                    display_label = get_doc_display_name(pdf_file_name)

                    with open(file_path, "rb") as file:
                        file_data = file.read()

                    st.download_button(
                        label=f"⬇️ {display_label}",
                        data=file_data,
                        file_name=pdf_file_name,
                        mime="application/pdf"
                    )
            else:
                st.info(T("sidebar_info_no_pdfs"))

        except Exception as e:
            st.error(f"Error listing or serving files: {e}")
    else:
        st.warning(T("sidebar_warning_dir"))

    st.markdown("---")
    st.markdown(f"LLM: `{LLM_MODEL}` (Temp: {LLM_TEMPERATURE})")
    st.markdown(f"Embeddings: `text-embedding-3-small` ({DIMENSION} dim)")
    st.markdown(T("sidebar_check_api"))


# --- Core RAG Logic (Cached for Initialization) ---

# Re-initialize the chain when k_retrieval or the selected_files list changes
@st.cache_resource(show_spinner=T("spinner_thinking"))
def initialize_rag_chain(index_name, k_retrieval, selected_files, llm_temperature):
    """
    Initializes Pinecone connection, creates the vector store retriever with filtering,
    and sets up the ConversationalRetrievalChain with system instructions.
    """
    if not selected_files:
        st.error(T("error_no_doc_selected"))
        return None

    try:
        if not all([os.getenv("OPENAI_API_KEY"), os.getenv("PINECONE_API_KEY"), os.getenv("PINECONE_ENVIRONMENT")]):
            st.error(T("error_missing_keys"))
            return None

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
        )

        metadata_filter = {"source": {"$in": selected_files}}

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": k_retrieval,
                "filter": metadata_filter
            }
        )

        llm = ChatOpenAI(
            temperature=llm_temperature,
            model_name=LLM_MODEL,
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        st.success(T("status_success"))
        return chain

    except Exception as e:
        st.error(T("error_rag_setup", e=e))
        return None


# --- Main Application Flow ---

# We only pass language-independent parameters to the cached function
rag_chain = initialize_rag_chain(
    INDEX_NAME,
    k_retrieval,
    st.session_state.selected_files,
    LLM_TEMPERATURE
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    avatar = "⚖️" if message["role"] == "assistant" else "user"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander(T("source_expander")):
                for j, source in enumerate(message["sources"]):
                    page_number_raw = source.metadata.get('page', 'N/A')
                    source_filename = source.metadata.get('source', 'Unknown Source')

                    # Use helper function to get localized display name
                    display_name = get_doc_display_name(source_filename)

                    try:
                        display_page = int(float(page_number_raw))
                    except (ValueError, TypeError):
                        display_page = page_number_raw

                    st.text_area(
                        T("source_label", j=j + 1, display_name=display_name, display_page=display_page),
                        value=source.page_content,
                        height=100,
                        key=f"source_view_{i}_{j}",
                        disabled=True
                    )

# Handle user input
if prompt := st.chat_input(T("chat_input_placeholder")):
    if rag_chain:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner(T("spinner_thinking")):
                try:
                    history_for_chain = [
                        (st.session_state.chat_history[k - 1]["content"], st.session_state.chat_history[k]["content"])
                        for k in range(1, len(st.session_state.chat_history), 2)
                    ]

                    result = rag_chain.invoke(
                        {"question": prompt, "chat_history": history_for_chain},
                    )

                    response_text = result["answer"]
                    sources = result.get("source_documents", [])

                    # ADD WARNING DISCLAIMER HERE
                    warning_text = T("warning_disclaimer")
                    full_response = f"{warning_text}\n\n{response_text}"

                    st.markdown(full_response)

                    if sources:
                        with st.expander(T("source_expander")):
                            for j, source in enumerate(sources):
                                page_number_raw = source.metadata.get('page', 'N/A')
                                source_filename = source.metadata.get('source', 'Unknown Source')

                                # Use helper function to get localized display name
                                display_name = get_doc_display_name(source_filename)

                                try:
                                    display_page = int(float(page_number_raw))
                                except (ValueError, TypeError):
                                    display_page = page_number_raw

                                st.text_area(
                                    T("source_label", j=j + 1, display_name=display_name, display_page=display_page),
                                    value=source.page_content,
                                    height=100,
                                    key=f"source_view_current_display_{j}",
                                    disabled=True
                                )

                    assistant_message = {
                        "role": "assistant",
                        "content": full_response,  # Store the full response with the warning
                        "sources": sources
                    }
                    st.session_state.chat_history.append(assistant_message)

                except Exception as e:
                    error_msg = T("error_generation", e=e)
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    print(f"RAG Error: {e}")

    else:
        st.error(T("error_chain_fail"))

# Clear chat button
if st.button(T("button_new_chat")):
    st.session_state["chat_history"] = []
    st.rerun()
