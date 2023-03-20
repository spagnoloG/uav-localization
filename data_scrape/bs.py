LAT_START = 45.9887
LAT_END = 46.1339
LNG_START = 14.4431
LNG_END = 14.5910

TILTS = [i for i in range(10, 60, 20)]
HEADINGS = [i for i in range(0, 360, 90)]

import numpy as np

## 3
## 4
#driver.save_screenshot('screenshot.png')
## 5
#driver.quit()

def generate_coordinates():
    urls = []
    latitudes = np.arange(LAT_START, LAT_END, 0.001)
    longitudes = np.arange(LNG_START, LNG_END, 0.001)
    for latitude in latitudes:
        for longitude in longitudes:
            for tilt in TILTS:
                params=f'?lng={longitude}&lat={latitude}&tilt={tilt}&heading=0'
                url = f'http://localhost:8000/index.html{params}'
                urls.append(url)

    return urls


urls = generate_coordinates()

# Write url line by line to text file

with open("urls.txt", "w") as output:
    for url in urls:
        output.write(url + "\n")

    output.close()
