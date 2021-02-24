from sentence_transformers import SentenceTransformer
import pca
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

def loadModel(model_name):
    try:
        with open('models/' + model_name + '.pickle', 'rb') as f:
            model = pickle.load(f)
            return model
    except:
        print('Downloading model')
    try:
        model = SentenceTransformer(model_name)
        with open('models/' + model_name + '.pickle', 'wb') as f:
            pickle.dump(model, f)
        return model
    except:
        print('Model does not exist')
        return None

def reduce3D(embeddings, model, anchors_pre_emb):

    anchors_post_emb = model.encode(anchors_pre_emb)
    data = np.append(embeddings, np.array(anchors_post_emb), axis=0)
    data_frame = pd.DataFrame(data)
    pca_model = pca(n_components=3)
    dict = pca_model.fit_transform(df)
    dim3 = np.array(dict['PC'])
    anchors = np.array([dim3[-3:]])
    return dim3, anchors

def getProjMatr(anchors):

    A, B, C = anchors
    B = B - A
    C = C - A
    Z = np.cross(B, C)
    M = np.array([B, Z]).T
    M = np.dot(np.dot(M, np.linalg.inv(np.dot(M.T, M))), M.T)
    return A, B, C, M, Z

def reduce2D(dim3, anchors):

    A, B, C, M, Z = getProjMatr(anchors)
    O = np.zeros(3)
    cent = np.array([i - A for i in dim3])
    flat = np.array([np.dot(M, i) for i in cent])
    dist = np.array([np.linalg.norm(cent_i - flat_i) for cent_i, flat_i in zip(cent, flat)])
    x = -(B / np.linalg.norm(B))
    y = Z / np.linalg.norm(Z)
    proj = np.array([np.array([np.dot(i, x), np.dot(i, y)]) for i in flat])
    return proj, dist

def projToPic(proj, dist, dim3, circ_size, img_w, border_px, alpha, image_name):

    def translate(a, min_a, max_a, min_b, max_b):
        range_a = max_a - min_a
        range_b = max_b - min_b
        return min_b + ((a - min_a) * range_b / range_a)

    sort_idx = np.argsort(dist)
    sorted_dist = np.flip(dist[sort_idx], axis=0)
    sorted_proj = np.flip(proj[sort_idx], axis=0)
    sorted_dim3 = np.flip(dim3[sort_idx], axis=0)

    max_x = max(proj.T[0])
    min_x = min(proj.T[0])
    max_y = max(proj.T[1])
    min_y = min(proj.T[1])

    range_x = max_x - min_x
    range_y = max_y - min_y
    img_asp_rat = range_y / range_x
    img_h = int(img_w * img_asp_rat)
    max_iw = img_w - border_px
    max_ih = img_h - border_px

    max_r = max(dim3.T[0])
    min_r = min(dim3.T[0])
    max_g = max(dim3.T[1])
    min_g = min(dim3.T[1])
    max_b = max(dim3.T[2])
    min_b = min(dim3.T[2])

    max_size = max(dist)
    min_size = min(dist)

    base = Image.new('RGBA', (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(base)
    for coords, size, color_data in zip(sorted_proj, sorted_dist, sorted_dim3):
        r, g, b = [ int(translate(color_data[0], min_r, max_r, 0, 255)),
                    int(translate(color_data[0], min_r, max_r, 0, 255)),
                    int(translate(color_data[0], min_r, max_r, 0, 255)),]
        x, y = coords
        trans_x = translate(x, min_x, max_x, border_px, max_iw)
        trans_y = translate(y, min_y, max_y, max_ih, border_px)
        trans_size = translate(size, min_size, max_size, 2, circ_size)
        bound = [   (trans_x - trans_size, trans_y - trans_size), 
                    (trans_x + trans_size, trans_y + trans_size)]
        draw.ellipse(bound, fill=(r, g, b, alpha))

    for coords in proj[-3:]:
        x, y = coords
        ol_c = (0, 0, 0)
        size = img_w * 0.015
        x = translate(x, min_x, max_x, border_px, max_iw)
        y = translate(y, min_y, max_y, max_ih, border_px)
        bound = [(x - size, y - size), (x + size, y + size)]
        draw.ellipse(bound, outline=ol_c, width=size/5)

    base.save(image_name + '.png')
    base.show()
    








