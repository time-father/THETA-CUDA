#!/usr/bin/env python3
"""Read and convert Theta's flat flux output.

Inputs:
  xml: Required positional path to a Theta input XML file.
  --kind: Select "flux" or "flux_std"; defaults to "flux".
  --input: Optional input flux path; defaults to the XML output path with the
           executable's "_gpu" suffix applied.
  --output: Optional CSV output path; defaults to the input stem with a .csv
            suffix, or a _decoded.csv suffix when the input is already CSV.

Output:
  A CSV matrix with one mesh region per row and one energy group per column.
  No output file is created if parsing or shape validation fails.
"""

import argparse
import array
import csv
import pathlib
import sys
import xml.etree.ElementTree as ET


def parse_args():
    """Parse the XML path, flux selection, and optional file overrides.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=pathlib.Path, help="Theta input XML file")
    parser.add_argument(
        "--kind", choices=("flux", "flux_std"), default="flux",
        help="XML output field to read (default: flux)")
    parser.add_argument(
        "--input", type=pathlib.Path,
        help="Override the input flux file path")
    parser.add_argument(
        "--output", type=pathlib.Path,
        help="Override the output CSV path")
    return parser.parse_args()


def read_xml_metadata(xml_path):
    """Read mesh dimensions, energy groups, and output encoding from XML.

    Args:
        xml_path: Path to a Theta input XML file.

    Returns:
        A tuple of (region count, group count, binary flag, precision,
        output node).

    Raises:
        ValueError: If the XML lacks required material, model, or output data.
    """
    root = ET.parse(xml_path).getroot()
    materials = root.find("materials")
    model = root.find("model")
    output = root.find("output")
    if materials is None or model is None or output is None:
        raise ValueError("XML must contain materials, model, and output nodes")

    groups_text = materials.findtext("ng")
    if groups_text is None:
        raise ValueError("XML materials node is missing ng")
    groups = int(groups_text.strip())

    region_count = 1
    dimension_count = 0
    for axis_name in ("x", "y", "z"):
        axis = model.find(axis_name)
        if axis is None:
            continue
        resolutions_text = axis.findtext("resolutions")
        if resolutions_text is not None:
            axis_regions = sum(int(value) for value in resolutions_text.split())
        else:
            edges_text = axis.findtext("edges")
            width_text = axis.findtext("width")
            if edges_text is not None:
                axis_regions = len(edges_text.split()) - 1
            elif width_text is not None:
                axis_regions = len(width_text.split())
            else:
                raise ValueError(
                    f"XML model axis {axis_name!r} needs edges or width")
        if axis_regions <= 0:
            raise ValueError(f"XML model axis {axis_name!r} has no regions")
        region_count *= axis_regions
        dimension_count += 1

    if dimension_count == 0 or groups <= 0:
        raise ValueError("XML must define at least one mesh axis and energy group")

    binary_text = (output.findtext("binary") or "false").strip().lower()
    binary = binary_text == "true"
    binary_node = output.find("binary")
    precision = "double"
    if binary_node is not None:
        precision = binary_node.attrib.get("precision", precision).lower()
    if precision not in ("float", "double"):
        raise ValueError("Binary precision must be float or double")

    return region_count, groups, binary, precision, output


def with_gpu_suffix(path):
    """Apply the output suffix used by the Theta executable.

    Args:
        path: Flux filename read from the XML output node.

    Returns:
        The filename with _gpu inserted before .csv/.bin, or appended otherwise.
    """
    if path.suffix in (".csv", ".bin"):
        return path.with_name(path.stem + "_gpu" + path.suffix)
    return pathlib.Path(str(path) + "_gpu")


def resolve_paths(args, output_node):
    """Resolve default input and CSV paths from command-line and XML settings.

    Args:
        args: Parsed command-line arguments.
        output_node: XML output element containing flux filenames.

    Returns:
        A tuple of (input path, output CSV path).

    Raises:
        ValueError: If the selected output filename is absent from the XML.
    """
    input_path = args.input
    if input_path is None:
        configured_name = output_node.findtext(args.kind)
        if configured_name is None or not configured_name.strip():
            raise ValueError(f"XML output node is missing {args.kind}")
        input_path = with_gpu_suffix(pathlib.Path(configured_name.strip()))

    output_path = args.output
    if output_path is None:
        if input_path.suffix.lower() == ".csv":
            output_path = input_path.with_name(input_path.stem + "_decoded.csv")
        else:
            output_path = input_path.with_suffix(".csv")
    return input_path, output_path


def read_values(path, binary, precision, expected_count, groups):
    """Read flux values and verify the expected flat region/group shape.

    Args:
        path: Input flux file path.
        binary: Whether the XML specifies raw binary output.
        precision: Binary scalar type, either float or double.
        expected_count: Expected number of region/group values.
        groups: Number of groups per output row.

    Returns:
        An array.array containing the flux values in region-major order.

    Raises:
        ValueError: If the file length or text values do not match the XML shape.
    """
    typecode = "f" if precision == "float" else "d"
    values = array.array(typecode)
    if binary:
        raw = path.read_bytes()
        if len(raw) % values.itemsize != 0:
            raise ValueError("Binary flux file size is not aligned to its precision")
        values.frombytes(raw)
    else:
        with path.open("r", newline="", encoding="utf-8") as flux_file:
            for row in csv.reader(flux_file):
                for value in row:
                    if value.strip():
                        values.append(float(value))

    if len(values) != expected_count:
        raise ValueError(
            f"Flux file has {len(values)} values; XML requires {expected_count} "
            f"({expected_count // groups} regions x {groups} groups)")
    return values


def write_csv(path, values, groups):
    """Write one mesh region per CSV row and one energy group per column.

    Args:
        path: Destination CSV file path.
        values: Flat region-major array of flux values.
        groups: Number of group values in each region row.

    Returns:
        None. Creates parent directories and writes the CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for offset in range(0, len(values), groups):
            writer.writerow(values[offset:offset + groups])


def main():
    """Convert the selected Theta flux output to a shape-checked CSV matrix.

    Returns:
        Process exit code: zero on success, nonzero on input or conversion error.
    """
    args = parse_args()
    try:
        regions, groups, binary, precision, output_node = read_xml_metadata(
            args.xml)
        input_path, output_path = resolve_paths(args, output_node)
        if input_path.resolve() == output_path.resolve():
            raise ValueError("Input and output paths must be different")
        values = read_values(
            input_path, binary, precision, regions * groups, groups)
        write_csv(output_path, values, groups)
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"read_flux.py: {error}", file=sys.stderr)
        return 2

    print(f"Read {regions} regions x {groups} groups from {input_path}")
    print(f"Wrote CSV matrix to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
