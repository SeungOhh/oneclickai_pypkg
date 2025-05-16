#%%
import numpy as np

# box to model output type
def encode(boxes, num_classes, high_stride, low_stride, area_threshold):

    # 각 해상도별 그리드 셀 크기 계산
    depth = 5 + num_classes  # 4 좌표 + 1 objectness + num_classes
    
    # 각 해상도별 라벨 텐서 초기화
    high_label = np.zeros((high_stride, high_stride, depth), dtype=np.float32)
    low_label  = np.zeros((low_stride, low_stride, depth), dtype=np.float32)
    
    # 박스가 하나도 없는 경우 빈 라벨 텐서 반환
    if boxes is None:
        return high_label, low_label

    # 각 박스에 대해 라벨 텐서에 할당
    for box in boxes:
        class_id = int(box[0])
        x = box[1]
        y = box[2]
        w = box[3]
        h = box[4]
        area = w * h  # normalized area

        # if area is too small, then continue
        if area < 0.0003:
            continue
        
        if area < area_threshold:
            # 작은 객체 → 높은 해상도 head 전용
            cell_x = int(x * high_stride) # 그리드 셀 x 좌표
            cell_y = int(y * high_stride) # 그리드 셀 y 좌표
            offset_x = x * high_stride - cell_x # 그리드 셀 내 x 좌표
            offset_y = y * high_stride - cell_y # 그리드 셀 내 y 좌표

            # 이미 할당된 객체가 있다면, 더 큰 영역의 박스로 덮어쓰기
            if high_label[cell_y, cell_x, 4] == 1.0:
                existing_w = high_label[cell_y, cell_x, 2]
                existing_h = high_label[cell_y, cell_x, 3]
                existing_area = existing_w * existing_h
                if area < existing_area:
                    continue  # 이미 더 큰 박스가 할당되어 있으면 건너뜀

            high_label[cell_y, cell_x, 0] = offset_x
            high_label[cell_y, cell_x, 1] = offset_y
            high_label[cell_y, cell_x, 2] = w
            high_label[cell_y, cell_x, 3] = h
            high_label[cell_y, cell_x, 4] = 1.0  # objectness
            high_label[cell_y, cell_x, 5:] = 0.0   # 기존 클래스 one-hot 초기화
            high_label[cell_y, cell_x, 5 + class_id] = 1.0
            
        else:
            # 큰 객체 → 낮은 해상도 head 전용
            cell_x = int(x * low_stride)
            cell_y = int(y * low_stride)
            offset_x = x * low_stride - cell_x
            offset_y = y * low_stride - cell_y

            if low_label[cell_y, cell_x, 4] == 1.0:
                existing_w = low_label[cell_y, cell_x, 2]
                existing_h = low_label[cell_y, cell_x, 3]
                existing_area = existing_w * existing_h
                if area < existing_area:
                    continue

            low_label[cell_y, cell_x, 0] = offset_x
            low_label[cell_y, cell_x, 1] = offset_y
            low_label[cell_y, cell_x, 2] = w
            low_label[cell_y, cell_x, 3] = h
            low_label[cell_y, cell_x, 4] = 1.0
            low_label[cell_y, cell_x, 5:] = 0.0
            low_label[cell_y, cell_x, 5 + class_id] = 1.0

    return high_label, low_label








# Example usage
if __name__ == "__main__":

    boxes_example = [
        [0.0, 0.076,0.076,0.7,0.7],
        [3.0, 0.53671875, 0.3578125, 0.0859375, 0.109375], 
        [0.0, 0.49921875, 0.93984375, 0.9296875, 0.2390625]
    ]

    high_label, low_label = encoded = encode(boxes_example, num_classes=5, high_stride=23, low_stride=3, area_threshold=0.08)
    print("High resolution label shape:", high_label.shape)
    print("Low resolution label shape:", low_label.shape)

