
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


iou = 0.6
depth = 3


# In[ ]:


df = pd.read_csv("../csv/test.csv")


# In[ ]:


def add_cluster_members(index, df, cluster_number, depth, iou):
    row = df.loc[[index]]
    for i in range(depth):
        candidate_rows = df[(df['frame_number'] == row.iloc[0]['frame_number'] + i + 1) & 
                           (df['class'] == row.iloc[0]['class'])]
        if not candidate_rows.empty:
            row_iou = calculate_highest_iou(row, candidate_rows)
            if row_iou and row_iou['iou'] >= iou:
                df.loc[row_iou['row'].index[0], 'cluster'] = cluster_number
                add_cluster_members(row_iou['row'].index[0], df, cluster_number, depth, iou)
                break

def calculate_highest_iou(row, candidate_rows):
    row_iou = {}
    iou_candidate = 0
    for i in range(candidate_rows.shape[0]):
        candidate = candidate_rows.iloc[[i]]
        if iou_candidate < calculate_iou(row, candidate):
            iou_candidate = calculate_iou(row, candidate)
            row_iou['row'] = candidate
            row_iou['iou'] = iou_candidate
    return row_iou

def calculate_iou(row, candidate):
    boxA = string_to_list(row.iloc[0]['box'])
    boxB = string_to_list(candidate.iloc[0]['box'])

    #Following code was copied from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def string_to_list(box):
    box = box[1:][:-1].split()
    return [float(i) for i in box]


# In[ ]:


df_clusters.loc[0, 'cluster'] = -1
df_clusters.loc[0, 'cluster']


# In[ ]:


df_clusters = df.copy()
df_clusters['cluster'] = -1

for i in range(df_clusters.shape[0]):
    if df_clusters.loc[i, 'cluster'] == -1:
        df_clusters.loc[i, 'cluster'] = i
        add_cluster_members(i, df_clusters, i, depth, iou)


# In[ ]:


df_clusters

