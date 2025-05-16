import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Fine-grained Hallucination Detection", layout="wide")

st.title("Fine-grained Hallucination Detection Example")

# First Example
st.header("Example 1")

# Reference Data
st.subheader("Reference Data")
reference_data = [
    ["tracking units", "20,326", "20,558"],
    ["Plant equipment and tooling", "13,078", "16,842"],
    ["Office equipment, computers and furniture", "11,553", "14,206"],
    ["Software", "31,349", "31,427"],
    ["Fixed assets not yet in service", "5,836", "4,657"],
]

# Convert to DataFrame for better display
df = pd.DataFrame(reference_data, columns=["Description", "2019", "2018"])
st.dataframe(df, hide_index=True)

# LM Output
st.subheader("LM Output")
lm_output = """
The change in fixed assets not yet in service from 2018 to 2019 was an increase of $1,280,500 ($5,836,000 - $4,657,000).
"""
st.text_area("", lm_output, height=100)

# Edited Version
st.subheader("Edited Version")
edited_output = """
The change in fixed assets not yet in service from 2018 to 2019 was an increase of <numerical><mark>$1,179,000</mark><delete>$1,280,500</delete></numerical> ($5,836,000 - $4,657,000).
"""

# Custom styling for the edited version
st.markdown("""
<style>
    .edited-text {
        font-family: monospace;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .deleted-text {
        color: red;
        text-decoration: line-through;
    }
    .marked-text {
        color: green;
    }
    .numerical-tag {
        font-weight: bold;
        color: #000000;
    }
    .contradictory-tag {
        font-weight: bold;
        color: #000000;
    }
    .entity-tag {
        font-weight: bold;
        color: #000000;
    }
    .relation-tag {
        font-weight: bold;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Process the edited output to apply styling
def process_edited_text(text):
    # Replace delete tags
    text = text.replace("<delete>", '<span class="deleted-text">')
    text = text.replace("</delete>", '</span>')
    
    # Replace mark tags
    text = text.replace("<mark>", '<span class="marked-text">')
    text = text.replace("</mark>", '</span>')
    
    # Replace numerical tags with bold styling
    text = text.replace("<numerical>", '<span class="numerical-tag">&lt;numerical&gt;</span>')
    text = text.replace("</numerical>", '</span>')
    
    # Replace contradictory tags with bold styling
    text = text.replace("<contradictory>", '<span class="contradictory-tag">&lt;contradictory&gt;</span>')
    text = text.replace("</contradictory>", '<span class="contradictory-tag">&lt;/contradictory&gt;</span>')
    
    # Replace entity tags with bold styling
    text = text.replace("<entity>", '<span class="entity-tag">&lt;entity&gt;</span>')
    text = text.replace("</entity>", '<span class="entity-tag">&lt;/entity&gt;</span>')
    
    # Replace relation tags with bold styling
    text = text.replace("<relation>", '<span class="relation-tag">&lt;relation&gt;</span>')
    text = text.replace("</relation>", '<span class="relation-tag">&lt;/relation&gt;</span>')
    
    return text

# Display the first example's edited version
st.markdown(f'<div class="edited-text">{process_edited_text(edited_output)}</div>', unsafe_allow_html=True)

# Second Example
st.header("Example 2")

# Reference Data
st.subheader("Reference Data")
reference_data2 = [
    ["Accrued Interest  ", "163", "1,598"],
    ["Accrued Expenses  ", "11,569", "7,362"],
    ["Settlement Deferred Compensation Liabilities  ", "3,830", "-"],
    ["Total as of December 31,  ", "15,562", "8,960"]
]

# Convert to DataFrame for better display
df2 = pd.DataFrame(reference_data2, columns=["Description", "2019", "2018"])
st.dataframe(df2, hide_index=True)

# LM Output
st.subheader("LM Output")
lm_output2 = """
The respective accrued interest in 2018 and 2019 are $1,598 and $163. The accrued expenses in 2019 were $5,000 higher than in 2018.
"""
st.text_area("", lm_output2, height=100)

# Edited Version
st.subheader("Edited Version")
edited_output2 = """
The respective accrued interest in 2018 and 2019 are $1,598 and $163. <contradictory><delete>The accrued expenses in 2019 were $5,000 higher than in 2018.</delete></contradictory>
"""

# Display the second example's edited version
st.markdown(f'<div class="edited-text">{process_edited_text(edited_output2)}</div>', unsafe_allow_html=True)

# Third Example
st.header("Example 3")

# Reference Data
st.subheader("Reference Data")
reference_text = """
the primary savings were experienced in employee salary and related costs through personnel reductions and reduced overhead costs from the sale of the landscaping business. as a result, earnings from service operations increased from $32.8 million for the year ended december 31, 2000, to $ 35.1 million for the year ended december 31, 2001
"""
st.text_area("", reference_text, height=150)

# LM Output
st.subheader("LM Output")
lm_output3 = """
The earnings from retail operations decreased from $32.8 million in 2000 to $35.1 million in 2001.
"""
st.text_area("", lm_output3, height=100)

# Edited Version
st.subheader("Edited Version")
edited_output3 = """
The earnings from <entity><mark>service operations</mark><delete>retail operations</delete></entity> <relation><mark>increased</mark><delete>decreased</delete></relation> from $32.8 million in 2000 to $35.1 million in 2001.
"""

# Display the third example's edited version
st.markdown(f'<div class="edited-text">{process_edited_text(edited_output3)}</div>', unsafe_allow_html=True) 
