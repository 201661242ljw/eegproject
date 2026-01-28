{ pkgs, ... }: {
  # To learn more about how to configure your Nix environment, see
  # https://developers.google.com/idx/guides/customize-idx-env
  channel = "stable-23.11"; # Or "unstable"
  packages = [
    pkgs.python3
    pkgs.pkg-config
    pkgs.SDL2
    pkgs.SDL2_image
    pkgs.SDL2_mixer
    pkgs.SDL2_ttf
    pkgs.libjpeg
    pkgs.libpng
    pkgs.freetype
    pkgs.portmidi
  ];
  env = {};
  services = {};
}
