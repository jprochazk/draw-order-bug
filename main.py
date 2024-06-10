#!/usr/bin/env python3

"""OCR-based inventory parser for Warframe."""

# Code by https://github.com/pajlada

from __future__ import annotations

import dataclasses
import itertools
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import easyocr
import Levenshtein
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from dataclass_wizard import JSONWizard


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class Item(JSONWizard):
    item_name: str


def find_closest_match(
    input_string: str, possible_strings: list[str]
) -> tuple[Optional[str], float]:
    closest_match = None
    min_distance = float("inf")  # Initialize with infinity

    for string in possible_strings:
        distance = Levenshtein.distance(input_string, string)
        if distance < min_distance:
            min_distance = distance
            closest_match = string

    return closest_match, min_distance


box_start_x = 102
box_start_y = 267
box_end_x = 324
box_end_y = 489
box_height = box_end_x - box_start_x
box_width = box_end_y - box_start_y
box_quantity_x_offset = 43
box_quantity_y_offset = 8
box_quantity_width = 80
box_quantity_height = 33
box_name_x_offset = 0
box_name_y_offset = 100
box_name_width = box_width
box_name_height = box_height - box_name_y_offset
box_right_margin = 59
box_bottom_margin = 45  # pixels between bottom of box above & below it


def cleanup_image(img):
    hsv = img
    mask1 = cv2.inRange(hsv, (100, 164, 185), (110, 175, 195))  # type: ignore
    mask2 = cv2.inRange(hsv, (93, 152, 171), (100, 163, 181))  # type: ignore
    mask3 = cv2.inRange(hsv, (40, 0, 171), (100, 255, 181))  # type: ignore
    mask = cv2.bitwise_or(mask1, mask2, mask3)
    target = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("/tmp/mask.png", target)
    rgb = target
    cv2.imwrite("/tmp/mask.png", rgb)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb = cv2.threshold(rgb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return rgb


class BoxCoords:
    start_x: int
    start_y: int

    @staticmethod
    def at(x: int, y: int) -> BoxCoords:
        self = BoxCoords()
        self.start_x = box_start_x + (x * (box_width + box_right_margin))
        self.start_y = box_start_y + (y * (box_height + box_bottom_margin))
        return self

    def full(self, sct, monitor_number):
        mon = sct.monitors[monitor_number]
        print(mon)
        return {
            "left": mon["left"] + self.start_x,
            "top": mon["top"] + self.start_y,
            "width": box_width,
            "height": box_height,
            "mon": monitor_number,
        }

    def full_rerun_mins(self):
        return [self.start_x, self.start_y]

    def full_rerun_sizes(self):
        return [box_width, box_height]

    def quantity_rerun_mins(self):
        return [
            self.start_x + box_quantity_x_offset,
            self.start_y + box_quantity_y_offset,
        ]

    def quantity_rerun_sizes(self):
        return [box_quantity_width, box_quantity_height]

    def quantity_slice(self, cv_img):
        x, y = self.quantity_rerun_mins()
        width, height = self.quantity_rerun_sizes()
        end_x = x + width
        end_y = y + height
        print(f"{x}-{end_x}, {y}-{end_y}")
        return cv_img[y:end_y, x:end_x]

    def name_slice(self, cv_img):
        x, y = self.name_rerun_mins()
        width, height = self.name_rerun_sizes()
        end_x = x + width
        end_y = y + height
        print(f"{x}-{end_x}, {y}-{end_y}")
        return cv_img[y:end_y, x:end_x]

    def quantity(self, sct, monitor_number):
        mon = sct.monitors[monitor_number]
        return {
            "left": mon["left"] + self.start_x,
            "top": mon["top"] + self.start_y,
            "width": box_quantity_width,
            "height": box_quantity_height,
            "mon": monitor_number,
        }

    def name_rerun_mins(self):
        return [self.start_x + box_name_x_offset, self.start_y + box_name_y_offset]

    def name_rerun_sizes(self):
        return [box_name_width, box_name_height]

    def name(self, sct, monitor_number):
        mon = sct.monitors[monitor_number]
        return {
            "left": mon["left"] + self.start_x,
            "top": mon["top"] + self.start_y,
            "width": box_name_width,
            "height": box_name_height,
            "mon": monitor_number,
        }


def easyocr_read(reader, clean_img) -> str:
    return " ".join([s[1] for s in reader.readtext(clean_img)])


def save_inventory2(con) -> None:
    with open("warframe-items-list.json", encoding="utf8") as fh:
        input_json_data = fh.read()
    items = Item.from_json(input_json_data)
    assert isinstance(items, list)
    item_names = [f.item_name for f in items]
    parsed_items: dict[str, int] = {}
    reader = easyocr.Reader(["en"])
    raw_img = cv2.imread("full-inventory.png")
    cv_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_BGRA2BGR)
    rr.log("image", rr.Image(cv_img, draw_order=10.0))
    for x, y in itertools.product(range(6), range(4)):
        box = BoxCoords.at(x, y)
        print(f"{x}/{y}")
        rr.log(
            f"image/{x}/{y}/full",
            rr.Boxes2D(mins=box.full_rerun_mins(), sizes=box.full_rerun_sizes()),
        )

        cropped_img = box.quantity_slice(cv_img)
        clean_img = cropped_img
        quantity = 1
        result = easyocr_read(reader, clean_img)
        print(result)
        try:
            quantity = int(result, 10)
        except:  # noqa: E722
            if result:
                print(f"error converting '{result}' to quantity")
                time.sleep(10)

        rr.log(
            f"image/{x}/{y}/quantity",
            rr.Boxes2D(
                mins=box.quantity_rerun_mins(),
                sizes=box.quantity_rerun_sizes(),
                labels=f"{quantity} ({result})",
            ),
        )
        rr.log(f"image/{x}/{y}/quantity/cleaned", rr.Image(clean_img, draw_order=5.0))
        rr.log(
            f"image/{x}/{y}/quantity/cleaned",
            rr.Transform3D(translation=[*box.quantity_rerun_mins(), 0]),
        )

        cropped_img = box.name_slice(cv_img)
        clean_img = cropped_img
        item_name = easyocr_read(reader, clean_img)
        nearest_item_name, _ = find_closest_match(item_name, item_names)
        if nearest_item_name:
            print(f"{nearest_item_name} = {quantity}")
            parsed_items[nearest_item_name] = quantity
        else:
            print(f"error parsing {item_name}")
            time.sleep(10)
            continue
        rr.log(
            f"image/{x}/{y}/name",
            rr.Boxes2D(
                mins=box.name_rerun_mins(),
                sizes=box.name_rerun_sizes(),
                labels=f"{nearest_item_name} ({item_name})",
            ),
        )
        print(result)

        con.execute(
            "INSERT INTO items (name, quantity) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET quantity = ?",
            (nearest_item_name, quantity, quantity),
        )
        con.commit()


def main() -> None:
    blueprint = rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial2DView(name="Image", origin="/", contents="image/**"),
        )
    )
    rr.init("warframe-save-inventory", spawn=True, default_blueprint=blueprint)

    con = sqlite3.connect("inventory.db")
    con.execute(
        "CREATE TABLE IF NOT EXISTS items (name TEXT PRIMARY KEY, quantity INTEGER)"
    )
    save_inventory2(con)
    con.close()


if __name__ == "__main__":
    main()
