import tensorflow as tf
import gradio as gr
import numpy as np

modelis = tf.keras.models.load_model("cnn_geriausias_modelis.keras")

klasiu_pavadinimai = ['Obelų rauplės (Apple scab)', 'Obelų juodasis vėžys (Black rot / Frogeye leaf spot)', 'Obelų rūdys (Cedar apple rust)', 'Sveikas lapas (Healthy)']

ligu_informacija = {'Obelų rauplės': {'pozymiai': 'Ant lapų atsiranda alyvuogių žalumo, vėliau tamsėjančių dėmių.', 'rekomendacija': 'Surinkite nukritusius pažeistus lapus, genėkite lają, kad geriau cirkuliuotų oras.'}, 'Obelų juodasis vėžys': {'pozymiai': 'Ant lapų gali atsirasti apvalių rusvų dėmių su tamsesniais kraštais.', 'rekomendacija': 'Pašalinkite pažeistas ir nudžiūvusias šakas, surinkite užkrėstus augalų likučius.'}, 'Obelų rūdys': {'pozymiai': 'Ant lapų susidaro ryškios geltonos ar oranžinės dėmės.', 'rekomendacija': 'Stebėkite ligos plitimą, pašalinkite stipriai pažeistus lapus, įvertinkite netoliese augančius kadagius.'}, 'Sveikas lapas': {'pozymiai': 'Ryškių ligos požymių nenustatyta.', 'rekomendacija': 'Toliau reguliariai stebėkite obels būklę.'}}

def pateikti_informacija(prognozes):
    if not prognozes:
        return "Įkelkite obels lapo nuotrauką."

    liga = max(prognozes, key=prognozes.get)
    ivertinimas = prognozes[liga]

    trumpas_pavadinimas = liga.split(" (")[0]
    informacija = ligu_informacija[trumpas_pavadinimas]

    tekstas = (
        f"### {liga}\n\n"
        f"**Modelio įvertinimas:** {ivertinimas:.1%}\n\n"
        f"**Požymiai:** {informacija['pozymiai']}\n\n"
        f"**Rekomendacija:** {informacija['rekomendacija']}\n\n"
        "⚠️ DI prognozė yra preliminari ir nepakeičia "
        "specialisto nustatytos diagnozės."
    )

    if ivertinimas < 0.70:
        tekstas += (
            "\n\n**Dėmesio:** modelio įvertinimas mažesnis "
            "nei 70 %. Rekomenduojama papildomai patikrinti lapą."
        )

    return tekstas


def atpazinti_liga(nuotrauka):
    if nuotrauka is None:
        return {}, "Įkelkite obels lapo nuotrauką."

    vaizdas = tf.image.resize(nuotrauka, [128, 128])
    vaizdas = tf.cast(vaizdas, tf.float32) / 255.0
    vaizdas = tf.expand_dims(vaizdas, axis=0)

    prognozes = modelis.predict(vaizdas, verbose=0)[0]


    prognoziu_zodynas = {
        pavadinimas: float(tikimybe)
        for pavadinimas, tikimybe in zip(
            klasiu_pavadinimai, prognozes
        )
    }

    aprasymas = pateikti_informacija(prognoziu_zodynas)

    return prognoziu_zodynas, aprasymas





programa = gr.Interface(
    fn=atpazinti_liga,
    inputs=gr.Image(
        type="numpy",
        label="Obels lapo nuotrauka"
    ),
    outputs=[
        gr.Label(
            num_top_classes=4,
            label="Atpažinimo rezultatai"
        ),
        gr.Markdown()
    ],
    title="Obelų lapų ligų atpažinimas",
    description=(
        "Įkelkite obels lapo nuotrauką ir sužinokite, "
        "ar lapas sveikas, ar turi ligos požymių.\n\n"
        "**Nuotraukos paruošimas:**\n\n"
        "- Padėkite vieną obels lapą ant vienspalvio, neutralaus paviršiaus.\n"
        "- Fotografuokite esant geram, tolygiam apšvietimui.\n"
        "- Nuotraukoje turi aiškiai matytis visas lapas ir jo pažeidimai.\n"
        "- Venkite ryškių šešėlių ir pašalinių objektų."
    ),
    flagging_mode="never",
    theme=gr.themes.Soft(
        font_mono=["Arial", "sans-serif"],
        primary_hue=gr.themes.Color(
            c50="#F2F5F2",
            c100="#E4EBE5",
            c200="#CCDACC",
            c300="#A9C1AE",
            c400="#7D9E86",
            c500="#52775F",
            c600="#315C46",
            c700="#294D3B",
            c800="#203D30",
            c900="#172D23",
            c950="#102018"
        )
    ).set(
        button_primary_background_fill="#315C46",
        button_primary_background_fill_hover="#244735",
        button_primary_text_color="#FFFFFF",
        color_accent_soft="#E4EDE6"
    )
)

if __name__ == "__main__":
    programa.launch()
