import logging.config

import streamlit as st
from dotenv import load_dotenv


load_dotenv()


# LOGGING_PATH = ROOT / "reactor/api/logging.conf"
#
# logging.config.fileConfig(LOGGING_PATH, disable_existing_loggers=False)
# logging.info(f"Loading logging config from: {LOGGING_PATH}")

logger = logging.getLogger(__name__)
logger.info("Creating Streamlit application")


if __name__ == "__main__":

    st.set_page_config(
        page_title="Home",
        layout="wide",
    )

    st.write("# Welcome ! 👋")
