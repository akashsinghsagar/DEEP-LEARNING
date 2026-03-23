import streamlit as st

# Page configuration
st.set_page_config(
    page_title="RNN Notebook Runner",
    page_icon="🔁",
    layout="centered"
)

st.title("🔁 RNN - One Section Runner")
st.markdown("**Runs the same RNN logic from your notebook in one section.**")

st.markdown("---")
st.subheader("Run RNN Code")

w_input = st.number_input("w_input", value=0.5, step=0.1, format="%.2f")
w_memory = st.number_input("w_memory", value=0.8, step=0.1, format="%.2f")
bias = st.number_input("bias", value=0.0, step=0.1, format="%.2f")

if st.button("▶ Run RNN", type="primary", use_container_width=True):
    sentence = ["I", "love", "deep", "learning"]

    word_value = {
        "I": 1,
        "love": 2,
        "deep": 3,
        "learning": 4
    }

    # convert words to numbers
    sequence = [word_value[word] for word in sentence]

    lines = []
    lines.append(f"Input sequence: {sentence}")
    lines.append(f"Encode sequence: {sequence}")
    lines.append("")
    lines.append("Processing sequence through RNN:")

    # STEP 3: Initialize the memory state
    memory = 0.0

    # STEP 4: Process the input sequence through the RNN
    for t in range(len(sequence)):
        input_value = sequence[t]
        new_memory = w_input * input_value + w_memory * memory + bias

        lines.append(f"Time step {t+1}:")
        lines.append(f"  Input value: {input_value}")
        lines.append(f"  Previous memory: {memory:.2f}")
        lines.append(f"  New memory: {new_memory:.2f}")
        lines.append("New Memory Calculation:")
        lines.append(
            f"  w_input * input_value: {w_input} * {input_value} = {w_input * input_value:.2f}"
        )

        memory = new_memory
        lines.append(f"  Updated memory: {memory:.2f}")
        lines.append("")

    lines.append(f"Final memory state after processing the sequence: {memory}")
    if memory > 5:
        lines.append("The model predicts a positive outcome.")
    else:
        lines.append("Bad")

    st.success("RNN run completed")
    st.text("\n".join(lines))

st.caption("Tip: Keep defaults to match your notebook output exactly.")
