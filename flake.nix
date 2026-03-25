{
  description = "Giggle ML";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    nixpkgs-old.url = "github:NixOS/nixpkgs/nixos-21.05";
    flake-utils.url = "github:numtide/flake-utils";
    giggle.url = "path:/Users/simon/Code/lab/giggle-dev/giggle";
    seqpare-src = {
      url = "github:deepstanding/seqpare";
      flake = false;
    };
  };

  outputs =
    {
      nixpkgs,
      nixpkgs-old,
      flake-utils,
      seqpare-src,
      giggle,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pkgs-old = nixpkgs-old.legacyPackages.${system};

        seqpare = pkgs.stdenv.mkDerivation {
          pname = "seqpare";
          version = "unstable";
          src = seqpare-src;

          buildInputs = with pkgs; [
            gnumake
            gcc
            zlib
          ];

          buildPhase = ''
            make --version
            make
          '';

          installPhase = ''
            mkdir -p $out/bin
            cp bin/seqpare $out/bin/
          '';
        };

        liftOver = pkgs.stdenv.mkDerivation rec {
          pname = "liftOver";
          version = "latest";

          src =
            if pkgs.stdenv.isDarwin then
              pkgs.fetchurl {
                url = "https://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/liftOver";
                sha256 = "sha256-udvT0UjLTBalZ8RxjRhmZUG+sJnnZcUK01yNWngG2Hg=";
              }
            else
              pkgs.fetchurl {
                url = "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver";
                sha256 = "sha256-0000000000000000000000000000000000000000000="; # Placeholder
              };

          # Since we are fetching a bare binary, we need to skip the unpack phase
          dontUnpack = true;

          # For Linux, we often need to patch the binary to find the right libraries
          nativeBuildInputs = pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.autoPatchelfHook ];
          buildInputs = pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.zlib
            pkgs.openssl
            pkgs.libpng
          ];

          installPhase = ''
            mkdir -p $out/bin
            cp $src $out/bin/liftOver
            chmod +x $out/bin/liftOver
          '';
        };
      in
      {
        devShells.default = pkgs.mkShellNoCC {
          packages = with pkgs; [
            seqpare
            uv
            just
            bedtools
            # wget
            # samtools
            htslib # bgzip
            giggle.packages.${system}.default
            liftOver
          ];
        };
      }
    );
}
