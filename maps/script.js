    var map, directionsService, directionsRenderer;

    function initMap() {

      // Objetos do Mapa
      var atores = [{
          type: "driver",
          position: {
            lat: -27.599440,
            lng: -48.673914
          },
          finalPosition: {
            lat: -27.597132,
            lng: -48.678086
          },
          img: "images/truck.png",
          name: "Guilherme R.",
          pontos: 1232
        },
        {
          type: "ccr",
          position: {
            lat: -27.587549,
            lng: -48.693284
          },
          img: "images/ccr.png"
        },
        {
          type: "friend",
          position: {
            lat: -27.593642,
            lng: -48.689540
          },
          finalPosition: {
            lat: -27.595016,
            lng: -48.683509
          },
          img: "https://cdn.discordapp.com/avatars/580890652611575818/207b87526cb635d548c54524f38c27cf.png?size=32",
          name: "Guilherme R.",
          pontos: 1232
        },
        {
          type: "friend",
          position: {
            lat: -27.605508,
            lng: -48.667739
          },
          finalPosition: {
            lat: -27.606830,
            lng: -48.664739
          },
          img: "https://cdn.discordapp.com/avatars/223456269103661056/0041b44d07706401e784a13e8fe1a742.png?size=32",
          name: "Guilherme R.",
          pontos: 1232
        },
        {
          type: "friend",
          position: {
            lat: -27.590081,
            lng: -48.681964
          },
          finalPosition: {
            lat: -27.594349,
            lng: -48.682980
          },
          img: "https://cdn.discordapp.com/avatars/481266461701046273/8d25c721b591fbd0b3f218684a89ff31.png?size=32",
          name: "Guilherme R.",
          pontos: 1232
        },
        {
          type: "friend",
          position: {
            lat: -27.600789,
            lng: -48.623471
          },
          finalPosition: {
            lat: -27.603112,
            lng: -48.631696
          },
          img: "https://cdn.discordapp.com/avatars/235512505210765323/58adb7fd58583c3c212e2a1c7fd2e381.png?size=32",
          name: "Guilherme R.",
          pontos: 1232
        },
        {
          type: "gastation",
          position: {
            lat: -27.610631,
            lng: -48.653053
          },
          img: "images/gas.png",
          name: "Posto 1",
          address: "SC 281",
          hour: "Aberto",
          quality: "9,8",
          price: "2,658"
        },
        {
          type: "gastation",
          position: {
            lat: -27.610416,
            lng: -48.652287
          },
          img: "images/gas.png",
          name: "Posto 1",
          address: "SC 281",
          hour: "Aberto",
          quality: "9,8",
          price: "2,658"
        },
        {
          type: "gastation",
          position: {
            lat: -27.601455,
            lng: -48.670139
          },
          img: "images/gas.png",
          name: "Posto 1",
          address: "SC 281",
          hour: "Aberto",
          quality: "9,8",
          price: "2,658"
        },
        {
          type: "gastation",
          position: {
            lat: -27.583736,
            lng: -48.697315
          },
          img: "images/gas.png",
          name: "Posto 1",
          address: "SC 281",
          hour: "Aberto",
          quality: "9,8",
          price: "2,658"
        }
      ];


      // Estilo do Mapa
      var style = [
        { elementType: 'geometry', stylers: [{ color: '#242f3e' }] },
        { elementType: 'labels.text.stroke', stylers: [{ color: '#242f3e' }] },
        { elementType: 'labels.text.fill', stylers: [{ color: '#746855' }] },
        {
          featureType: 'administrative.locality',
          elementType: 'labels.text.fill',
          stylers: [{ color: '#d59563' }]
        },
        {
          featureType: 'poi',
          elementType: 'labels.text.fill',
          stylers: [{ color: '#d59563' }]
        },
        {
          featureType: 'poi.park',
          elementType: 'geometry',
          stylers: [{ color: '#263c3f' }]
        },
        {
          featureType: 'poi.park',
          elementType: 'labels.text.fill',
          stylers: [{ color: '#6b9a76' }]
        },
        {
          featureType: 'road',
          elementType: 'geometry',
          stylers: [{ color: '#38414e' }]
        },
        {
          featureType: 'road',
          elementType: 'geometry.stroke',
          stylers: [{ color: '#212a37' }]
        },
        {
          featureType: 'road',
          elementType: 'labels.text.fill',
          stylers: [{ color: '#9ca5b3' }]
        },
        {
          featureType: 'road.highway',
          elementType: 'geometry',
          stylers: [{ color: '#746855' }]
        },
        {
          featureType: 'road.highway',
          elementType: 'geometry.stroke',
          stylers: [{ color: '#1f2835' }]
        },
        {
          featureType: 'road.highway',
          elementType: 'labels.text.fill',
          stylers: [{ color: '#f3d19c' }]
        },
        {
          featureType: 'transit',
          elementType: 'geometry',
          stylers: [{ color: '#2f3948' }]
        },
        {
          featureType: 'transit.station',
          elementType: 'labels.text.fill',
          stylers: [{ color: '#d59563' }]
        },
        {
          featureType: 'water',
          elementType: 'geometry',
          stylers: [{ color: '#17263c' }]
        },
        {
          featureType: 'water',
          elementType: 'labels.text.fill',
          stylers: [{ color: '#515c6d' }]
        },
        {
          featureType: 'water',
          elementType: 'labels.text.stroke',
          stylers: [{ color: '#17263c' }]
        }
      ];


      // Criar o Mapa
      map = new google.maps.Map(document.getElementById('map'), {
        center: { lat: atores[0].position.lat, lng: atores[0].position.lng },
        zoom: 13,
        disableDefaultUI: true,
        zoomControl: true,
        styles: style
      });

      // Traçar a Rota Inicial e Final e esconder Marks
      directionsService = new google.maps.DirectionsService();
      directionsRenderer = new google.maps.DirectionsRenderer({ suppressMarkers: true, preserveViewport: true });
      directionsRenderer.setMap(map);
      var onChangeHandler = function() {
        calculateAndDisplayRoute(directionsService, directionsRenderer, "Continente Shopping", "Parque Aquático Arco Iris");
      };
      onChangeHandler();

      // Mostrar os Objetos no Mapa
      var infowindow = new google.maps.InfoWindow();

      var marker, i;
      var markers = [];


      for (i = 0; i < atores.length; i++) {
        marker = new google.maps.Marker({
          position: new google.maps.LatLng(atores[i].position.lat, atores[i].position.lng),
          map: map,
          icon: atores[i].img
        });

        markers.push(marker);

        if (atores[i].type == "friend") {

          google.maps.event.addListener(marker, 'click', (function(marker, i) {
            return function() {
              infowindow.setContent(
                $(".friend_box").html()
                .replace("{nome}", atores[i].name)
                .replace("{pontos}", atores[i].pontos)
              );
              infowindow.open(map, marker);
            }
          })(marker, i));
        }
        if (atores[i].type == "gastation") {

          google.maps.event.addListener(marker, 'click', (function(marker, i) {
            return function() {
              infowindow.setContent(
                $(".gastation_box").html()
                .replace("{nome}", atores[i].name)
                .replace("{endereco}", atores[i].address)
                .replace("{horario}", atores[i].hour)
                .replace("{qualidade}", atores[i].quality)
                .replace("{precodiesel}", atores[i].price)
              );
              infowindow.open(map, marker);
            }
          })(marker, i));

        }
        // Você está aqui
        if (atores[i].type == "driver") {

          google.maps.event.addListener(marker, 'click', (function(marker, i) {
            return function() {
              infowindow.setContent(
                "Você está aqui!"
              );
              infowindow.open(map, marker);
            }
          })(marker, i));

        }
        // Simular Caminho
        if (atores[i].type == "driver" || atores[i].type == "friend") {

          var passos = 400;
          var delay = 100;

          var timeinterval = setInterval(function(marker, i) {
            if (atores[i].step == undefined) {
              atores[i].step = 0;
            }
            atores[i].step++;
            if (atores[i].step == passos) {
              for (i = 0; i < 100; i++) {
                window.clearInterval(i);
              }
            }

            var latinicial = atores[i].position.lat;
            var latfinal = atores[i].finalPosition.lat;
            var latatual = latinicial + ((latfinal - latinicial) / passos) * atores[i].step;
            var lnginicial = atores[i].position.lng;
            var lngfinal = atores[i].finalPosition.lng;
            var lngatual = lnginicial + ((lngfinal - lnginicial) / passos) * atores[i].step;
            var novaposicao = new google.maps.LatLng(latatual, lngatual);
            marker.setPosition(novaposicao)

          }, delay, marker, i);
        }
      }
    }

    // Traçar Rota
    function calculateAndDisplayRoute(directionsService, directionsRenderer, from, to) {
      directionsService.route({
          origin: { query: from },
          destination: { query: to },
          travelMode: 'DRIVING'
        },
        function(response, status) {
          if (status === 'OK') {
            directionsRenderer.setDirections(response);
          }
          else {
            window.alert('Directions request failed due to ' + status);
          }
        });
    }

    function handleLocationError(browserHasGeolocation, infoWindow, pos) {
      infoWindow.setPosition(pos);
      infoWindow.setContent(browserHasGeolocation ?
        'Error: The Geolocation service failed.' :
        'Error: Your browser doesn\'t support geolocation.');
      infoWindow.open(map);
    }

    var i = 0;
    var direcoes = [
      { text: "Subida em 300m comece a acelerar", audio: "audios/audio_1.mp3" },
      { text: "Opa! Parece que você está acelerando demais, vá com calma", audio: "audios/audio_3.mp3" },
      { text: "Seu freio tá sofrendo! Vem suave!", audio: "audios/audio_4.mp3" }
    ];

    $(".direcao").click(function() {
      $(".direcao_text").html(direcoes[i].text);
      var audio = new Audio(direcoes[i].audio);
      audio.play();
      i++;
    });