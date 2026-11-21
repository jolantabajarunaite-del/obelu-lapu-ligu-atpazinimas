import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="Obelų lapų būklės vertinimas", page_icon="🍃", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background: #F8FAF7; color: #26372E; }
.block-container { max-width: 1100px; padding-top: 2.4rem; padding-bottom: 3rem; }
h1, h2, h3 { color: #203F31 !important; letter-spacing: -.025em; }
h1 { font-size: clamp(2rem, 4vw, 3.1rem) !important; font-weight: 650 !important; }
h2 { font-size: 1.55rem !important; }
p, li { line-height: 1.65; }
.eyebrow { font-size: .77rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; color: #547463; margin-bottom: .7rem; }
.hero-copy { font-size: 1.12rem; color: #54675B; max-width: 760px; line-height: 1.75; }
.hero-rule { height: 3px; width: 64px; background: #648C70; margin: 1.4rem 0 1.8rem; border-radius: 8px; }
.soft-panel { padding: 1.2rem 1.4rem; background: #EDF3EC; border: 1px solid #D9E5D9; border-radius: 12px; margin: .7rem 0 1.3rem; }
.result-panel { background: #EAF1EA; border-left: 4px solid #477454; border-radius: 10px; padding: 1.1rem 1.4rem; margin: .8rem 0 1.2rem; }
.result-kicker { font-size: .8rem; color: #58705F; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; }
.result-title { font-size: 1.55rem; font-weight: 650; color: #204A34; margin: .25rem 0; }
.result-score { color: #4B6655; font-size: .96rem; }
 .info-card {background:#FFFFFF;border:1px solid #E0E9DF;border-radius:14px;padding:1.2rem 1.35rem;min-height:145px;margin:.45rem 0 1rem;box-shadow:0 6px 22px rgba(26,58,38,.035)}
 .info-card h4 {margin:0 0 .45rem;color:#264734;font-size:1.03rem}
 .info-card p {margin:0;color:#506456;font-size:.94rem}
 .section-note {font-size:.93rem;color:#596D60;margin-bottom:1rem}
 .footer { border-top: 1px solid #DDE7DE; padding-top: 1.3rem; margin-top: 2.5rem; font-size: .86rem; color: #607266; }
[data-testid="stFileUploader"] { background: white; border: 1px solid #DCE8DE; border-radius: 12px; padding: .65rem; }
[data-testid="stProgressBar"] > div > div { background: #477454; }
</style>
""", unsafe_allow_html=True)

KLASES = [
    "Obelų rauplės",
    "Obelų juodasis vėžys",
    "Obelų rūdys",
    "Sveikas lapas",
]

# Klasių tvarka atitinka originalų modelį ir nekeičia jo prognozių.
LIGOS = {
    KLASES[0]: {
        "pozymiai": "Alyvuogių žalumo, vėliau rudos ar beveik juodos, kartais aksominės dėmės ant lapų. Gali būti pažeisti ir vaisiai.",
        "veiksmai": "Apžiūrėkite kitus lapus ir vaisius. Rudenį surinkite ir pašalinkite ligos pažeistus nukritusius lapus bei vaisius, o tinkamu genėjimo metu praretinkite tankią lają. Jei rauplės kartojasi, pavasarį įvertinkite profilaktinės apsaugos poreikį.",
    },
    KLASES[1]: {
        "pozymiai": "Apvalios rusvos dėmės, dažnai su tamsesniu violetiniu pakraščiu; kartais dėmės primena „varlės akį“. Liga pažeidžia ne tik lapus, bet ir vaisius bei medžio žievę – gali sukelti vaisių puvinį ir šakų vėžinius pažeidimus.",
        "veiksmai": "Apžiūrėkite vaisius ir šakas. Pašalinkite ant medžio likusius mumifikuotus vaisius ir negyvas ar pažeistas šakeles. Genėkite sausu oru, nepalikite infekuotų vaisių sode ir stenkitės nepažeisti žievės.",
    },
    KLASES[2]: {
        "pozymiai": "Ryškios geltonos ar oranžinės dėmės; kai kurių rūdžių atveju apatinėje lapo pusėje susidaro išaugos. Panašius požymius gali sukelti skirtingos rūdžių rūšys.",
        "veiksmai": "Įvertinkite ligos išplitimą. Jei pažeisti tik pavieniai lapai, stebėkite obels būklę. Jei rūdys kartojasi ir stipriai pažeidžia lapiją, apsaugą planuokite kitam pavasariui, prieš užsikrėtimą.",
    },
    KLASES[3]: {
        "pozymiai": "Lapas atrodo sveikas, matomų ligos požymių nėra.",
        "veiksmai": "Specialių apsaugos priemonių nereikia. Pakanka įprastos obels priežiūros.",
    },
}


# Sezoninė priežiūra ir cheminė apsauga, atskirai nuo diagnostinio modelio.
APSAUGA = {
    KLASES[0]: {
        "pavasaris": "Nuo pumpurų brinkimo ir lapų skleidimosi vertinkite rauplių infekcijos riziką, ypač drėgnu oru. Profilaktiniai fungicidai naudojami tik pagal etiketėje nurodytą obels augimo tarpsnį ir purškimo sąlygas.",
        "vasara": "Lietingu laikotarpiu gali kilti naujų rauplių infekcijų rizika. Papildomos apsaugos poreikį lemia oro sąlygos ir ligos paplitimas. Laikykitės etiketėje nurodytų purškimo normų, leidžiamo purškimų skaičiaus ir išlaukos laikotarpio.",
        "ruduo": "Surinkite ir pašalinkite nukritusius ligos pažeistus lapus bei vaisius. Tai mažina infekcijos šaltinius kitam sezonui; rudeninis viso medžio purškimas nėra privaloma universali procedūra.",
        "preparatai": "**Vitra 40 WG** – vario hidroksido pagrindu pagamintas fungicidas, skirtas profilaktinei obelų apsaugai nuo rauplių. Purškimo laiką, normą, leidžiamą purškimų skaičių ir išlaukos laikotarpį būtina pasirinkti pagal galiojančią produkto etiketę.",
        "saltiniai": [],
    },
    KLASES[1]: {
        "pavasaris": "Apžiūrėkite žievę ir šakas, pašalinkite nudžiūvusias bei vėžio pažeistas šakas. Jeigu liga patvirtinta ir kartojasi, dėl fungicidinės apsaugos spręskite pagal konkrečios ligos registruotą paskirtį.",
        "vasara": "Pašalinkite ant medžio likusius ligotus ar mumifikuotus vaisius. Vien purškimas nepašalina jau susidariusių šakų pažeidimų; venkite žievės sužalojimų.",
        "ruduo": "Surinkite ir pašalinkite pažeistus vaisius bei lapus, o nudžiūvusias ar pažeistas šakas išpjaukite tinkamu genėjimo metu. Tai mažina galimus infekcijos šaltinius.",
        "preparatai": "",
        "saltiniai": [],
    },
    KLASES[2]: {
        "pavasaris": "Pagrindinis profilaktinės apsaugos laikotarpis – nuo lapų skleidimosi iki laikotarpio po žydėjimo. Drėgnais orais, esant infekcijos rizikai, gali būti naudojami tik Lietuvoje nuo atitinkamų obelų rūdžių registruoti fungicidai.",
        "vasara": "Jeigu ant lapų jau matomos rūdžių dėmės, purškimas jų nepanaikins. Apsaugą nuo šios ligos tikslinga planuoti kitam pavasariui.",
        "ruduo": "Specialus rudeninis obelų purškimas nuo rūdžių nerekomenduojamas, nes šios rūdžių grupės sukėlėjai žiemoja ant kadagių.",
        "preparatai": "",
        "saltiniai": [],
    },
    KLASES[3]: {
        "pavasaris": "Jeigu ligų požymių nėra, o infekcijos rizika nedidelė, profilaktinis purškimas nebūtinai reikalingas.",
        "vasara": "Reguliariai apžiūrėkite lapus ir vaisius. Chemines priemones naudokite tik nustačius konkrečią problemą ir įvertinus poreikį.",
        "ruduo": "Surinkite ligų pažeistas augalų liekanas ir prižiūrėkite medį; vien dėl sveiko lapo prognozės rudeninis purškimas nereikalingas.",
        "preparatai": "Nenustačius ligos, konkretaus fungicido nerekomenduojame.",
        "saltiniai": [],
    },
}

VATIS_PAIESKA = "https://vatis.vatzum.lt/aapSarasas"

@st.cache_resource
def ikelti_modeli():
    return tf.keras.models.load_model("cnn_geriausias_modelis.keras")

st.title("Obelų lapų būklės vertinimas")
st.markdown('<div class="hero-copy">Įkelkite obels lapo nuotrauką ir sužinokite, ar lapas sveikas, ar turi požymių, būdingų vienai iš trijų atpažįstamų ligų: obelų rauplėms, juodajam vėžiui arba rūdims.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-rule"></div>', unsafe_allow_html=True)

kairys, desinys = st.columns([1.1, .9], gap="large")
with kairys:
    st.subheader("01 · Įkelkite nuotrauką")
    nuotrauka = st.file_uploader("Obels lapo nuotrauka", type=["jpg", "jpeg", "png"], help="JPG, JPEG arba PNG formatas")
    with st.expander("Kaip tinkamai nufotografuoti lapą?", expanded=False):
        st.markdown("""- Fotografuokite vieną lapą, kad matytųsi visas jo paviršius.
- Pasirinkite tolygų natūralų apšvietimą ir neutralų foną.
- Venkite ryškių šešėlių, atspindžių ir pašalinių objektų.
""")
with desinys:
    st.markdown('<div class="soft-panel"><strong>Ką gali įvertinti modelis?</strong><p>Obelų rauples, obelų juodąjį vėžį, obelų rūdis ir sveiką lapą.</p><p><strong>Svarbu:</strong> prognozė nėra laboratorinė diagnozė. Net aukštas procentas negarantuoja, kad liga atpažinta teisingai.</p></div>', unsafe_allow_html=True)

if nuotrauka is not None:
    try:
        vaizdas = Image.open(nuotrauka).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Nepavyko atidaryti nuotraukos. Pasirinkite tinkamą JPG arba PNG failą.")
        st.stop()
    st.divider()
    st.subheader("02 · Analizės rezultatas")
    kaire, desine = st.columns([.85, 1.15], gap="large")
    with kaire:
        st.image(vaizdas, caption="Jūsų įkelta nuotrauka", width="stretch")
    with desine:
        try:
            modelis = ikelti_modeli()
            masyvas = np.array(vaizdas)
            masyvas = tf.image.resize(masyvas, [128, 128])
            masyvas = tf.cast(masyvas, tf.float32) / 255.0
            masyvas = tf.expand_dims(masyvas, axis=0)
            prognozes = modelis.predict(masyvas, verbose=0)[0]
        except Exception:
            st.error("Nepavyko atlikti analizės. Patikrinkite, ar modelio failas pasiekiamas.")
            st.stop()
        indeksas = int(np.argmax(prognozes))
        liga = KLASES[indeksas]
        ivertinimas = float(prognozes[indeksas])
        st.markdown(f'<div class="result-panel"><div class="result-kicker">Vertinimo rezultatas</div><div class="result-title">{liga}</div><div class="result-score">Modelio priskirtas įvertinimas: {ivertinimas:.1%}</div></div>', unsafe_allow_html=True)
        if ivertinimas < .70:
            st.warning("Modelio įvertinimas mažesnis nei 70 %. Rezultatą būtina patikrinti papildomai. Ši riba nėra patvirtintas nežinomų ligų nustatymo metodas.")
        else:
            st.info("Net aukštas įvertinimas neatmeta kitos ligos, kenkėjų ar fiziologinio pažeidimo galimybės.")
        with st.expander("Išsamūs vertinimo rezultatai"):
            for pavadinimas, tikimybe in zip(KLASES, prognozes):
                st.write(f"{pavadinimas} · {float(tikimybe):.1%}")
                st.progress(float(tikimybe))
    st.subheader("03 · Rekomendacijos" if liga == KLASES[3] else "03 · Ligos valdymas ir apsauga")
    st.markdown('<div class="info-card"><h4>'+('Rekomendacija' if liga == KLASES[3] else 'Rekomenduojami veiksmai')+'</h4><p>'+LIGOS[liga]["veiksmai"]+'</p></div>', unsafe_allow_html=True)
    if liga != KLASES[3]:
        with st.expander("Cheminė apsauga: kada ir kuo purkšti?"):
            apsauga = APSAUGA[liga]
            for sezonas, raktas in [("Pavasarį", "pavasaris"), ("Vasarą", "vasara"), ("Rudenį", "ruduo")]:
                st.markdown(f"**{sezonas}**")
                st.write(apsauga[raktas])
            if apsauga["preparatai"]:
                st.markdown("**Registruoti preparatai**")
                st.markdown(apsauga["preparatai"])
            st.caption("Augalų apsaugos produktus naudokite tik pagal galiojančią Lietuvos registraciją ir etiketę: paskirtį, naudotojo kategoriją, normą, purškimo laiką, išlaukos laikotarpį ir saugos reikalavimus.")
            st.markdown(f"**Registruotų augalų apsaugos produktų paieška:** [VATIS – augalų apsaugos produktai]({VATIS_PAIESKA})")
            if apsauga["saltiniai"]:
                st.markdown("**Informacijos šaltiniai**")
                for saltinio_pavadinimas, saltinio_nuoroda in apsauga["saltiniai"]:
                    st.markdown(f"- [{saltinio_pavadinimas}]({saltinio_nuoroda})")

    with st.expander("Ką daryti, jei rezultatas kelia abejonių?"):
        st.write("""Programėlė skirta trims obelų lapų ligoms atpažinti ir sveikų lapų būklei įvertinti. Kitų ligų ar pažeidimų ji neatpažįsta.

Jei rezultatas kelia abejonių, rekomenduojama pasikonsultuoti su augalų apsaugos specialistu.""")

st.divider()
st.subheader("Pažinkite tris lapų ligas")
st.markdown('<div class="section-note">Šie požymiai padeda palyginti modelio rezultatą su tuo, ką matote ant lapo. Vien iš nuotraukos liga ne visada nustatoma patikimai.</div>', unsafe_allow_html=True)
for a, b, c in [(KLASES[0],KLASES[1],KLASES[2])]:
    k1,k2,k3=st.columns(3,gap="medium")
    for col, name in zip((k1,k2,k3),(a,b,c)):
        with col:
            st.markdown('<div class="info-card"><h4>'+name+'</h4><p>'+LIGOS[name]["pozymiai"]+'</p></div>',unsafe_allow_html=True)

with st.expander("Modelio galimybės ir ribotumai"):
    st.markdown("""**Modelis:** CNN, apmokytas 1 773 „PlantVillage“ obelų lapų nuotraukomis. Keturios klasės: rauplės, juodasis vėžys, rūdys ir sveikas lapas.

**Testavimo rezultatai:** 91,35 % tikslumas su atskirtomis „PlantVillage“ nuotraukomis (266 vaizdai). Papildomame bandyme su 120 „Plant Pathology 2021“ nuotraukų tikslumas buvo 25 %. Šie rezultatai rodo, kad modelio patikimumas priklauso nuo nuotraukų pobūdžio.

**Svarbu:** keturių klasių procentai yra modelio išvestys, o ne patvirtinta diagnozės tikimybė. Modelis nėra išmokytas atpažinti visas ligas, kenkėjų ar aplinkos sukeltus pažeidimus. Net ir didelis įvertinimas neatmeta klaidos.""")

st.subheader("Apie projektą")
st.write("Programėlė sukurta studijų projekto metu. Naudojamas konvoliucinis neuroninis tinklas (CNN), sukurtas su „TensorFlow“ ir „Keras“. Internetinė sąsaja sukurta naudojant „Streamlit“.")
with st.expander("Duomenų šaltiniai ir autorių teisės"):
    st.markdown("""Modeliui mokyti naudotos „PlantVillage“ obelų lapų nuotraukos. Duomenų rinkinio šaltinis: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset). Mokslinis šaltinis: Mohanty, Hughes ir Salathé (2016), *Using Deep Learning for Image-Based Plant Disease Detection*.

Duomenų rinkinio naudojimo sąlygos priklauso nuo konkrečios jo publikavimo versijos. Šaltinio ir autorių nurodymas nereiškia, kad trečiųjų šalių nuotraukų autorinės teisės perduodamos šios programėlės kūrėjams.""")
with st.expander("Privatumas"):
    st.markdown("""Įkeltos obelų lapų nuotraukos naudojamos tik jų būklei įvertinti. Programėlė nekaupia nuotraukų duomenų bazėje ir nenaudoja jų dirbtinio intelekto modeliui papildomai mokyti.

Nuotraukos apdorojamos programėlės veikimo metu. Programėlė veikia „Streamlit Community Cloud“ platformoje, kurios techninių duomenų tvarkymui taikoma paslaugos teikėjo privatumo politika.

Rekomenduojama įkelti tik augalų nuotraukas, kuriose nėra asmens duomenų.""")

st.markdown('<div class="footer">DI modelio rezultatai yra preliminarūs ir nepakeičia profesionalios augalų ligų diagnostikos.</div>', unsafe_allow_html=True)
