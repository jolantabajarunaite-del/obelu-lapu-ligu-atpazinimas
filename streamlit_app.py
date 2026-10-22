
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Obelų lapų ligų atpažinimas",
    page_icon="🍃",
    layout="centered"
)

st.markdown("""
<style>
h1, h2, h3 {
    color: #315C46;
}
.stButton > button {
    background-color: #315C46;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def ikelti_modeli():
    return tf.keras.models.load_model("cnn_geriausias_modelis.keras")

modelis = ikelti_modeli()

klasiu_pavadinimai = [
    "Obelų rauplės (Apple scab)",
    "Obelų juodasis vėžys (Black rot / Frogeye leaf spot)",
    "Obelų rūdys (Cedar apple rust)",
    "Sveikas lapas (Healthy)"
]

ligu_informacija = {
    "Obelų rauplės": {
        "pozymiai": "Ant lapų atsiranda alyvuogių žalumo, vėliau tamsėjančių dėmių.",
        "rekomendacija": "Surinkite nukritusius pažeistus lapus, genėkite lają, kad geriau cirkuliuotų oras."
    },
    "Obelų juodasis vėžys": {
        "pozymiai": "Ant lapų gali atsirasti apvalių rusvų dėmių su tamsesniais kraštais.",
        "rekomendacija": "Pašalinkite pažeistas ir nudžiūvusias šakas, surinkite užkrėstus augalų likučius."
    },
    "Obelų rūdys": {
        "pozymiai": "Ant lapų susidaro ryškios geltonos ar oranžinės dėmės.",
        "rekomendacija": "Stebėkite ligos plitimą, pašalinkite stipriai pažeistus lapus, įvertinkite netoliese augančius kadagius."
    },
    "Sveikas lapas": {
        "pozymiai": "Ryškių ligos požymių nenustatyta.",
        "rekomendacija": "Toliau reguliariai stebėkite obels būklę."
    }
}

st.title("🍃 Obelų lapų ligų atpažinimas")

st.write(
    "Įkelkite obels lapo nuotrauką ir sužinokite, "
    "ar lapas sveikas, ar turi ligos požymių."
)

st.markdown("""
**Nuotraukos paruošimas:**
- Padėkite vieną obels lapą ant vienspalvio, neutralaus paviršiaus.
- Fotografuokite esant geram, tolygiam apšvietimui.
- Nuotraukoje turi aiškiai matytis visas lapas ir jo pažeidimai.
- Venkite ryškių šešėlių ir pašalinių objektų.
""")

nuotrauka = st.file_uploader(
    "Pasirinkite obels lapo nuotrauką",
    type=["jpg", "jpeg", "png"]
)

if nuotrauka is not None:
    vaizdas = Image.open(nuotrauka).convert("RGB")
    st.image(vaizdas, caption="Įkelta nuotrauka")

    masyvas = np.array(vaizdas)
    masyvas = tf.image.resize(masyvas, [128, 128])
    masyvas = tf.cast(masyvas, tf.float32) / 255.0
    masyvas = tf.expand_dims(masyvas, axis=0)

    prognozes = modelis.predict(masyvas, verbose=0)[0]
    indeksas = int(np.argmax(prognozes))
    liga = klasiu_pavadinimai[indeksas]
    ivertinimas = float(prognozes[indeksas])

    st.subheader("Atpažinimo rezultatai")
    st.write(f"**{liga}**")
    st.write(f"**Modelio įvertinimas:** {ivertinimas:.1%}")

    for pavadinimas, tikimybe in zip(klasiu_pavadinimai, prognozes):
        st.write(pavadinimas)
        st.progress(float(tikimybe))

    trumpas = liga.split(" (")[0]
    informacija = ligu_informacija[trumpas]

    st.markdown(f"**Požymiai:** {informacija['pozymiai']}")
    st.markdown(f"**Rekomendacija:** {informacija['rekomendacija']}")

    if ivertinimas < 0.70:
        st.warning(
            "Modelio įvertinimas mažesnis nei 70 %. "
            "Rekomenduojama papildomai patikrinti lapą."
        )

    st.info(
        "DI prognozė yra preliminari ir nepakeičia "
        "specialisto nustatytos diagnozės."
    )
