from pathlib import Path

from medunda.downloader import configure_parser as downloader_parser
from medunda.downloader import downloader
from medunda.reducer import build_action_args
from medunda.reducer import configure_parser as reducer_parser
from medunda.reducer import reducer


def test_cmems_climatology_download(tmp_path: Path):
    parser = downloader_parser()
    d_args = parser.parse_args(
        [
            "create",
            "--variables",
            "chl",
            "thetao",
            "--start-date",
            "2000-01-01",
            "--end-date",
            "2000-12-31",
            "--frequency",
            "monthly",
            "--domain",
            "domains/GSA9.yaml",
            "--output-dir",
            str(tmp_path),
        ]
    )
    downloader(d_args)

    r_args = reducer_parser().parse_args(
        [
            "--input-dataset",
            str(tmp_path),
            "--output-file",
            str(tmp_path / "cmems_climatology.nc"),
            "climatology",
            "--variable",
            "chl",
            "--frequency",
            "monthly",
        ]
    )

    dataset_path = r_args.input_dataset
    variables = r_args.variables
    output_file = r_args.output_file
    format = r_args.format
    action_name = r_args.action

    reducer(
        dataset_path=dataset_path,
        output_file=output_file,
        format=format,
        action_name=action_name,
        variables=variables,
        args=build_action_args(r_args),
    )

    # TODO: add checks on the output file
    assert True
