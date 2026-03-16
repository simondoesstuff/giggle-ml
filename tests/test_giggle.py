"""Tests for giggleml.data.giggle module."""

from unittest.mock import MagicMock, patch

import pytest

from giggleml.data.giggle import GiggleIndex, GiggleResult, _check_tool_in_path


class TestCheckToolInPath:
    """Tests for _check_tool_in_path helper."""

    def test_tool_exists(self):
        with patch("shutil.which", return_value="/usr/bin/giggle"):
            _check_tool_in_path("giggle")

    def test_tool_missing(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="giggle not found in PATH"):
                _check_tool_in_path("giggle")


class TestGiggleResult:
    """Tests for GiggleResult dataclass."""

    def test_all_fields_required(self):
        result = GiggleResult(
            file="test.bed.gz",
            file_size=10,
            overlaps=100,
            odds_ratio=2.5,
            fishers_two_tail=0.001,
            fishers_left_tail=1.0,
            fishers_right_tail=0.0005,
            combo_score=50.0,
        )
        assert result.file == "test.bed.gz"
        assert result.file_size == 10
        assert result.overlaps == 100
        assert result.odds_ratio == 2.5
        assert result.fishers_two_tail == 0.001
        assert result.fishers_left_tail == 1.0
        assert result.fishers_right_tail == 0.0005
        assert result.combo_score == 50.0


class TestGiggleIndex:
    """Tests for GiggleIndex class."""

    @pytest.fixture
    def mock_tools(self):
        """Mock giggle and bgzip as available in PATH."""
        with patch("shutil.which", return_value="/usr/bin/tool"):
            yield

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory with fake bed files."""
        bed_dir = tmp_path / "beds"
        bed_dir.mkdir()
        (bed_dir / "sample1.bed.gz").write_bytes(b"")
        (bed_dir / "sample2.bed.gz").write_bytes(b"")
        return bed_dir

    @pytest.fixture
    def significance_output(self):
        """Standard significance output for mocking."""
        return (
            "#file\tfile_size\toverlaps\todds_ratio\tfishers_two_tail\tfishers_left_tail\tfishers_right_tail\tcombo_score\n"
            "/path/to/sample.bed.gz\t50\t500\t2.5\t0.001\t1.0\t0.0005\t50.0\n"
        )

    def test_init_missing_tools(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="giggle not found"):
                GiggleIndex("/some/dir")

    def test_init_invalid_directory(self, mock_tools):
        with pytest.raises(ValueError, match="Directory does not exist"):
            GiggleIndex("/nonexistent/path")

    def test_init_valid_directory(self, mock_tools, temp_dir):
        index = GiggleIndex(temp_dir)
        assert index.directory == temp_dir
        assert index._index_dir_path == temp_dir.with_suffix(".giggle")

    def test_init_custom_index_dir(self, mock_tools, temp_dir, tmp_path):
        custom_index = tmp_path / "custom_index"
        index = GiggleIndex(temp_dir, index_dir=custom_index)
        assert index._index_dir_path == custom_index

    def test_init_sorted_parameter(self, mock_tools, temp_dir):
        index = GiggleIndex(temp_dir, sorted=True)
        assert index._sorted is True

    def test_init_genome_size_parameter(self, mock_tools, temp_dir):
        index = GiggleIndex(temp_dir, genome_size=3000000000)
        assert index._genome_size == 3000000000

    def test_exists_false(self, mock_tools, temp_dir):
        index = GiggleIndex(temp_dir)
        assert not index.exists

    def test_exists_true(self, mock_tools, temp_dir):
        temp_dir.with_suffix(".giggle").mkdir()
        index = GiggleIndex(temp_dir)
        assert index.exists

    def test_index_dir_creates_when_missing(self, mock_tools, temp_dir):
        index = GiggleIndex(temp_dir)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _ = index.index_dir

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "giggle"
            assert cmd[1] == "index"
            assert "-i" in cmd
            assert "-o" in cmd
            assert "-s" not in cmd

    def test_index_dir_with_sorted_flag(self, mock_tools, temp_dir):
        index = GiggleIndex(temp_dir, sorted=True)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _ = index.index_dir

            cmd = mock_run.call_args[0][0]
            assert "-s" in cmd

    def test_index_dir_skips_when_exists(self, mock_tools, temp_dir):
        temp_dir.with_suffix(".giggle").mkdir()
        index = GiggleIndex(temp_dir)
        with patch("subprocess.run") as mock_run:
            _ = index.index_dir
            mock_run.assert_not_called()

    def test_list_beds(self, mock_tools, temp_dir):
        temp_dir.with_suffix(".giggle").mkdir()
        index = GiggleIndex(temp_dir)

        giggle_output = (
            f"{temp_dir}/sample1.bed.gz\t1000\n{temp_dir}/sample2.bed.gz\t2000\n"
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=giggle_output, stderr=""
            )
            beds = index.list_beds

            assert beds == {"sample1.bed.gz", "sample2.bed.gz"}
            cmd = mock_run.call_args[0][0]
            assert "-l" in cmd

    def test_list_beds_caches_result(self, mock_tools, temp_dir):
        temp_dir.with_suffix(".giggle").mkdir()
        index = GiggleIndex(temp_dir)

        giggle_output = f"{temp_dir}/sample1.bed.gz\t1000\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=giggle_output, stderr=""
            )
            beds1 = index.list_beds
            beds2 = index.list_beds

            assert beds1 is beds2
            assert mock_run.call_count == 1

    def test_parse_output(self, mock_tools, temp_dir, significance_output):
        index = GiggleIndex(temp_dir)
        results = index._parse_output(significance_output)

        assert len(results) == 1
        assert results[0].file == "sample.bed.gz"
        assert results[0].file_size == 50
        assert results[0].overlaps == 500
        assert results[0].odds_ratio == 2.5
        assert results[0].fishers_two_tail == 0.001
        assert results[0].fishers_left_tail == 1.0
        assert results[0].fishers_right_tail == 0.0005
        assert results[0].combo_score == 50.0

    def test_parse_output_skips_header(self, mock_tools, temp_dir, significance_output):
        index = GiggleIndex(temp_dir)
        results = index._parse_output(significance_output)
        assert len(results) == 1

    def test_parse_output_skips_empty_lines(self, mock_tools, temp_dir):
        index = GiggleIndex(temp_dir)
        output = (
            "#header\n"
            "/path/to/sample.bed.gz\t50\t500\t2.5\t0.001\t1.0\t0.0005\t50.0\n"
            "\n\n"
        )
        results = index._parse_output(output)
        assert len(results) == 1

    def test_query_file_not_found(self, mock_tools, temp_dir):
        temp_dir.with_suffix(".giggle").mkdir()
        index = GiggleIndex(temp_dir)

        with pytest.raises(ValueError, match="Query file does not exist"):
            index.query("/nonexistent/query.bed")

    def test_query_with_gzipped_file(self, mock_tools, temp_dir, significance_output):
        temp_dir.with_suffix(".giggle").mkdir()
        query_file = temp_dir / "query.bed.gz"
        query_file.write_bytes(b"")
        index = GiggleIndex(temp_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=significance_output, stderr=""
            )
            results = index.query(query_file)

            assert len(results) == 1
            cmd = mock_run.call_args[0][0]
            assert "-q" in cmd
            assert "-s" in cmd
            assert str(query_file) in cmd

    def test_query_with_plain_bed_uses_bgzip(
        self, mock_tools, temp_dir, significance_output
    ):
        temp_dir.with_suffix(".giggle").mkdir()
        query_file = temp_dir / "query.bed"
        query_file.write_text("chr1\t100\t200\n")
        index = GiggleIndex(temp_dir)

        call_count = 0

        def mock_subprocess_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if cmd[0] == "bgzip":
                return MagicMock(returncode=0)
            return MagicMock(returncode=0, stdout=significance_output, stderr="")

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            with patch("builtins.open", MagicMock()):
                index.query(query_file)

        assert call_count == 2

    def test_query_uses_genome_size(self, mock_tools, temp_dir, significance_output):
        temp_dir.with_suffix(".giggle").mkdir()
        query_file = temp_dir / "query.bed.gz"
        query_file.write_bytes(b"")
        index = GiggleIndex(temp_dir, genome_size=3000000000)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=significance_output, stderr=""
            )
            index.query(query_file)

            cmd = mock_run.call_args[0][0]
            assert "-g" in cmd
            assert "3000000000" in cmd

    def test_self_query_not_in_index(self, mock_tools, temp_dir):
        temp_dir.with_suffix(".giggle").mkdir()
        index = GiggleIndex(temp_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=f"{temp_dir}/sample1.bed.gz\t1000\n", stderr=""
            )
            with pytest.raises(ValueError, match="not found in index"):
                index.self_query("nonexistent.bed.gz")

    def test_self_query_success(self, mock_tools, temp_dir, significance_output):
        temp_dir.with_suffix(".giggle").mkdir()
        index = GiggleIndex(temp_dir)

        list_output = f"{temp_dir}/sample1.bed.gz\t1000\n"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout=list_output, stderr=""),
                MagicMock(returncode=0, stdout=significance_output, stderr=""),
            ]
            results = index.self_query("sample1.bed.gz")

            assert len(results) == 1
            assert mock_run.call_count == 2

    def test_query_regions_single(self, mock_tools, temp_dir, significance_output):
        temp_dir.with_suffix(".giggle").mkdir()
        index = GiggleIndex(temp_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=significance_output, stderr=""
            )
            index.query_regions("chr1:100-200")

            cmd = mock_run.call_args[0][0]
            assert "-r" in cmd
            assert "chr1:100-200" in cmd
            assert "-s" in cmd

    def test_query_regions_multiple(self, mock_tools, temp_dir, significance_output):
        temp_dir.with_suffix(".giggle").mkdir()
        index = GiggleIndex(temp_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=significance_output, stderr=""
            )
            index.query_regions(["chr1:100-200", "chr2:300-400"])

            cmd = mock_run.call_args[0][0]
            r_index = cmd.index("-r")
            assert cmd[r_index + 1] == "chr1:100-200,chr2:300-400"

    def test_reindex_removes_existing(self, mock_tools, temp_dir):
        index_dir = temp_dir.with_suffix(".giggle")
        index_dir.mkdir()
        (index_dir / "some_file").write_text("data")
        index = GiggleIndex(temp_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            index.reindex()

            assert not (index_dir / "some_file").exists()
            mock_run.assert_called_once()

    def test_reindex_clears_cache(self, mock_tools, temp_dir):
        temp_dir.with_suffix(".giggle").mkdir()
        index = GiggleIndex(temp_dir)

        list_output1 = f"{temp_dir}/sample1.bed.gz\t1000\n"
        list_output2 = (
            f"{temp_dir}/sample1.bed.gz\t1000\n{temp_dir}/sample2.bed.gz\t2000\n"
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=list_output1, stderr=""
            )
            beds1 = index.list_beds
            assert len(beds1) == 1

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="", stderr=""),
                MagicMock(returncode=0, stdout=list_output2, stderr=""),
            ]
            index.reindex()
            temp_dir.with_suffix(".giggle").mkdir(exist_ok=True)
            beds2 = index.list_beds
            assert len(beds2) == 2
