from PIL import Image
import matplotlib.pylab as plt
import requests
import io
from struct import unpack
import math
import time

start = time.time()

url = 'https://reference.pictures/wp-content/uploads/2021/02/Reference-Pictures_Figure-Drawing_Chelsie-329-scaled.jpg'
r = requests.get(url)

with io.BytesIO(r.content) as f:
    with Image.open(f) as img:
        plt.figure()
        plt.imshow(img)
        plt.grid(False)
        plt.show()

"""## Rozkład struktury pliku"""

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}

data = r.content

huffman_data = []
quant_data = []
DCT_data = []
scan_data = []

file_len = len(data)
while (len(data) > 0):
    marker, = unpack(">H", data[0:2])
    offset = file_len - len(data)
    print(offset, "\t", marker_mapping.get(marker))
    if marker == 0xffd8:
        data = data[2:]
    elif marker == 0xffd9:
        break
    else:
        len_chunk, = unpack(">H", data[2:4])
        len_chunk += 2
        chunk = data[4:len_chunk]
        if marker == 0xFFC4:
            huffman_data.append(chunk)
        elif marker == 0xFFDB:
            quant_data.append(chunk)
        elif marker == 0xFFC0:
            DCT_data.append(chunk)
        elif marker == 0xffda:
            scan_data = data[len_chunk:]
            data = data[-2:]
            continue
        data = data[len_chunk:]

"""## Tabele kwantyzacji"""


def GetArray(type, l, length):
    """
    A convenience function for unpacking an array from bitstream
    """
    s = ""
    for i in range(length):
        s = s + type
    return list(unpack(s, l[:length]))


quant = {}

for data in quant_data:
    (hdr,) = unpack("B", data[0:1])
    quant[hdr] = GetArray("B", data[1: 1 + 64], 64)
    for x in range(8):
        print(quant[hdr][x:x + 8])
    print()

"""## Tabele Huffmana"""


class HuffmanTable:
    """
    A Huffman Table class
    """

    def __init__(self):
        self.root = []
        self.elements = []

    def BitsFromLengths(self, root, element, pos):
        if isinstance(root, list):
            if pos == 0:
                if len(root) < 2:
                    root.append(element)
                    return True
                return False
            for i in [0, 1]:
                if len(root) == i:
                    root.append([])
                if self.BitsFromLengths(root[i], element, pos - 1) == True:
                    return True
        return False

    def GetHuffmanBits(self, lengths, elements):
        self.elements = elements
        ii = 0
        for i in range(len(lengths)):
            for j in range(lengths[i]):
                self.BitsFromLengths(self.root, elements[ii], i)
                ii += 1

    def Find(self, st):
        r = self.root
        while isinstance(r, list):
            r = r[st.GetBit()]
        return r

    def GetCode(self, st):
        while True:
            res = self.Find(st)
            if res == 0:
                return 0
            elif res != -1:
                return res


huffman_tables = {}

for data in huffman_data:
    offset = 0
    (header,) = unpack("B", data[offset: offset + 1])
    offset += 1

    lengths = GetArray("B", data[offset: offset + 16], 16)
    offset += 16
    elements = []
    for i in lengths:
        elements += GetArray("B", data[offset: offset + i], i)
        offset += i

    print("Header: ", header & 0x0F, (header >> 4) & 0x0F)
    print("Lengths: ", lengths)
    print("Number of elements: ", len(elements))

    hf = HuffmanTable()
    hf.GetHuffmanBits(lengths, elements)
    huffman_tables[header] = hf

"""## Opis obrazu"""

quantMapping = []

for data in DCT_data:
    hdr, img_height, img_width, components = unpack(">BHHB", data[0:6])
    img_width = ((img_width + 7) // 8) * 8
    img_height = ((img_height + 7) // 8) * 8

    for i in range(components):
        id, samp, QtbId = unpack("BBB", data[6 + i * 3:9 + i * 3])
        quantMapping.append(QtbId)

    print("size %ix%i" % (img_width, img_height))
    print(quantMapping)

"""## Odczyt zawartości obrazu
### Usuwanie dodatkowych zer
"""


def RemoveFF00(data):
    datapro = []
    i = 0
    while (True):
        b, bnext = unpack("BB", data[i:i + 2])
        if i < 12:
            print(b, bnext)
        if (b == 0xff):
            if (bnext != 0):
                break
            datapro.append(data[i])
            i += 2
        else:
            datapro.append(data[i])
            i += 1
    return datapro, i


print(len(scan_data))
scan_data, lenchunk = RemoveFF00(scan_data)


class Stream:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def GetBit(self):
        b = self.data[self.pos >> 3]
        s = 7 - (self.pos & 0x7)
        self.pos += 1
        return (b >> s) & 1

    def GetBitN(self, l):
        val = 0
        for i in range(l):
            val = val * 2 + self.GetBit()
        return val


st = Stream(scan_data)

image = [0] * (img_width * img_height);

class IDCT:
    def __init__(self):
        self.base = [0] * 64
        self.zigzag = [
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
        ]
        self.idct_precision = 8
        self.idct_table = [
            [
                (self.NormCoeff(u) * math.cos(((2.0 * x + 1.0) * u * math.pi) / 16.0))
                for x in range(self.idct_precision)
            ]
            for u in range(self.idct_precision)
        ]

    def NormCoeff(self, n):
        if n == 0:
            return 1.0 / math.sqrt(2.0)
        else:
            return 1.0

    def rearrange_using_zigzag(self):
        for x in range(8):
            for y in range(8):
                self.zigzag[x][y] = self.base[self.zigzag[x][y]]
        return self.zigzag

    def perform_IDCT(self):
        out = [list(range(8)) for i in range(8)]

        for x in range(8):
            for y in range(8):
                local_sum = 0
                for u in range(self.idct_precision):
                    for v in range(self.idct_precision):
                        local_sum += (
                                self.zigzag[v][u]
                                * self.idct_table[u][x]
                                * self.idct_table[v][y]
                        )
                out[y][x] = local_sum // 4
        self.base = out


def DecodeNumber(code, bits):
    l = 2 ** (code - 1)
    if bits >= l:
        return bits
    else:
        return bits - (2 * l - 1)


def BuildMatrix(st, idx, quant, olddccoeff):
    i = IDCT()

    code = huffman_tables[0 + idx].GetCode(st)
    bits = st.GetBitN(code)
    dccoeff = DecodeNumber(code, bits) + olddccoeff

    i.base[0] = (dccoeff) * quant[0]
    l = 1
    while l < 64:
        code = huffman_tables[16 + idx].GetCode(st)
        if code == 0:
            break

        # The first part of the AC quantization table
        # is the number of leading zeros
        if code > 15:
            l += code >> 4
            code = code & 0x0F

        bits = st.GetBitN(code)

        if l < 64:
            coeff = DecodeNumber(code, bits)
            i.base[l] = coeff * quant[l]
            l += 1

    i.rearrange_using_zigzag()
    i.perform_IDCT()

    return i, dccoeff


def Clamp(col):
    col = 255 if col > 255 else col
    col = 0 if col < 0 else col
    return int(col)


def ColorConversion(Y, Cr, Cb):
    R = Cr * (2 - 2 * .299) + Y
    B = Cb * (2 - 2 * .114) + Y
    G = (Y - .114 * B - .299 * R) / .587
    return (Clamp(R + 128), Clamp(G + 128), Clamp(B + 128))


oldlumdccoeff, oldCbdccoeff, oldCrdccoeff = 0, 0, 0
for y in range(img_height // 8):
    for x in range(img_width // 8):
        matL, oldlumdccoeff = BuildMatrix(st, 0, quant[quantMapping[0]], oldlumdccoeff)
        matCr, oldCrdccoeff = BuildMatrix(st, 1, quant[quantMapping[1]], oldCrdccoeff)
        matCb, oldCbdccoeff = BuildMatrix(st, 1, quant[quantMapping[2]], oldCbdccoeff)
        for yy in range(8):
            for xx in range(8):
                image[(x * 8 + xx) + ((y * 8 + yy) * img_width)] = ColorConversion(matL.base[yy][xx],
                                                                                   matCb.base[yy][xx],
                                                                                   matCr.base[yy][xx])

"""## Wyświetlanie obrazu"""

img = Image.new("RGB", (img_width, img_height))
img.putdata(image)

plt.figure()
plt.imshow(img)
plt.grid(False)
plt.show()

end = time.time()
print("Total time ", end - start)

"""## Zadania
Napisz algorytm odczytujący obrazek zapisany w formacie JPEG. W tym celu uzupełnij powyższy kod o fragmenty, realizujące następujące zadania:
1. Odczyt zawartości obrazu z odwróceniem dwuwymiarowej transformacji kosinusowej włącznie. Wykorzystaj fakt, że dwuwymiarowa transformacja jest serią jednowymiarowych przekształceń. (3 pkt.)
2. Napisz szybką, tj. bez wykorzystywania pętli i z wcześniej obliczonymi współczynnikami, wersję ww. metody. (2 pkt.)

## Odnośniki
- https://yasoob.me/posts/understanding-and-writing-jpeg-decoder-in-python/
"""
