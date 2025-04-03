import os
import argparse
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def normalizeStaining(img, Io=240, alpha=1, beta=0.15):
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = img.shape
    img = img.reshape((-1, 3))

    OD = -np.log((img.astype(np.float32) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]

    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    return Inorm


def process_image(input_path, output_path):
    try:
        img = np.array(Image.open(input_path).convert("RGB"))
        norm_img = normalizeStaining(img)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(norm_img).save(output_path)
        return (input_path, "success")
    except Exception as e:
        return (input_path, f"failed: {e}")


def collect_images(root_dir):
    image_list = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(("png", "jpg", "jpeg", "tif", "tiff")):
                full_path = os.path.join(root, fname)
                image_list.append(full_path)
    return image_list


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    threads = args.threads

    image_files = collect_images(input_dir)
    log = []

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for input_path in image_files:
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            futures.append(executor.submit(process_image, input_path, output_path))

        for future in tqdm(as_completed(futures), total=len(futures), desc="归一化处理中"):
            result = future.result()
            log.append(result)

    log_path = os.path.join(output_dir, "normalization_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        for fname, status in log:
            f.write(f"{fname}\t{status}\n")

    print(f"✅ 所有图像处理完成，日志保存在 {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='原始图像文件夹')
    parser.add_argument('--output_dir', type=str, required=True, help='保存归一化图像的文件夹')
    parser.add_argument('--threads', type=int, default=4, help='并行线程数')
    args = parser.parse_args()

    main(args)
