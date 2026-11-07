# Obelų lapų ligų atpažinimas naudojant DI

Projektas skirtas obelų lapų ligoms atpažinti iš skaitmeninių nuotraukų, naudojant konvoliucinį neuroninį tinklą (CNN).

## Atpažįstamos klasės

- Obelų rauplės (Apple scab)
- Obelų juodasis vėžys (Black rot)
- Obelų rūdys (Cedar apple rust)
- Sveiki obelų lapai (Healthy)

## Duomenys ir modelis

Modelis sukurtas naudojant „PlantVillage“ duomenų rinkinį. Iš viso naudotos 1 773 obelų lapų nuotraukos.

Nuotraukos apdorotos 128 × 128 pikselių RGB formatu. Modelio testavimo tikslumas – 91,35 %.

## Naudotos technologijos

Python, TensorFlow, Keras, Streamlit.

## Programėlės naudojimas

1. Atidaryti internetinę programėlę.
2. Įkelti obels lapo nuotrauką.
3. Peržiūrėti modelio nustatytą klasę ir prognozės tikimybės įvertį.

Programėlė: https://obelu-lapu-ligu-atpazinimas.streamlit.app/

## Projekto ribotumai

Sistema yra demonstracinė ir skirta preliminariam obelų lapų būklės vertinimui. Modelis atpažįsta tik keturias numatytas klases, o jo tikslumas natūraliomis fotografavimo sąlygomis nėra pakankamai ištirtas.
