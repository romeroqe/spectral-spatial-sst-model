from pygam import GAM, l
from statsmodels.tsa.seasonal import seasonal_decompose as decompose

def gam_fit(X_pixel, y_pixel):
        n_activos = X_pixel.shape[1]

        terms = l(0)
        for k in range(1, n_activos):
            terms += l(k)

        gam_pixel = GAM(terms=terms).fit(X_pixel, y_pixel)
        return gam_pixel.predict(X_pixel)

def seasonal_decompose(serie):
    result = decompose(serie, model='additive', period=12)
    return result.seasonal, result.trend