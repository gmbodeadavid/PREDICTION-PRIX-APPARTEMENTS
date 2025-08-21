from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from django.views.decorators.http import require_http_methods
import pandas as pd
import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR.parent / 'prediction_Prix_Appartements.pkl'
DATASET_PATH = BASE_DIR.parent / 'Dataset_Propre.csv'


def _load_dataset_locations() -> list[str]:
    if DATASET_PATH.exists():
        dataset = pd.read_csv(DATASET_PATH)
        if 'location' in dataset.columns:
            return sorted(dataset['location'].dropna().unique().tolist())
    return []


def _load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, 'rb') as model_file:
            return pickle.load(model_file)
    return None


@require_http_methods(["GET"])
def home(request: HttpRequest) -> HttpResponse:
    locations = _load_dataset_locations()
    context = {
        'locations': locations,
    }
    return render(request, 'predictor/home.html', context)


@require_http_methods(["POST"])
def predict(request: HttpRequest) -> HttpResponse:
    model = _load_model()
    locations = _load_dataset_locations()

    if model is None:
        return render(request, 'predictor/home.html', {
            'locations': locations,
            'error': "Modèle introuvable. Assurez-vous que 'prediction_Prix_Appartements.pkl' est à la racine du projet.",
        })

    try:
        selected_location = request.POST.get('location') or ''
        total_sqft = float(request.POST.get('total_sqft') or 0)
        bath = int(float(request.POST.get('bath') or 0))
        balcony = int(float(request.POST.get('balcony') or 0))
        bedrooms = int(float(request.POST.get('bedrooms') or 0))

        input_df = pd.DataFrame([[selected_location, total_sqft, bath, balcony, bedrooms]],
                                columns=['location', 'total_sqft', 'bath', 'balcony', 'bedrooms'])

        prediction = model.predict(input_df)
        price_fcfa = round(float(prediction[0]) * 1000, 2)

        return render(request, 'predictor/result.html', {
            'price_fcfa': price_fcfa,
            'form_values': {
                'location': selected_location,
                'total_sqft': total_sqft,
                'bath': bath,
                'balcony': balcony,
                'bedrooms': bedrooms,
            },
        })
    except Exception as exc:
        return render(request, 'predictor/home.html', {
            'locations': locations,
            'error': f"Erreur lors de la prédiction: {exc}",
        })
