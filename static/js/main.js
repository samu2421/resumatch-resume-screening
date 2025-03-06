(function ($) {
    "use strict";

    // Spinner
    var spinner = function () {
        setTimeout(function () {
            if ($('#spinner').length > 0) {
                $('#spinner').removeClass('show');
            }
        }, 1);
    };
    spinner();
    
    // Execute spinner function when document is ready
    $(document).ready(function() {
        spinner();
        
        // Initiate the wowjs
        if (typeof WOW !== 'undefined') {
            new WOW().init();
        }
        
        // Sticky Navbar
        $(window).scroll(function () {
            if ($(this).scrollTop() > 45) {
                $('.navbar').addClass('sticky-top shadow-sm');
            } else {
                $('.navbar').removeClass('sticky-top shadow-sm');
            }
        });
        
        // Back to top button
        $(window).scroll(function () {
            if ($(this).scrollTop() > 100) {
                $('.back-to-top').fadeIn('slow');
            } else {
                $('.back-to-top').fadeOut('slow');
            }
        });
        
        $('.back-to-top').click(function () {
            $('html, body').animate({scrollTop: 0}, 1500, 'easeInOutExpo');
            return false;
        });
        
        // Skills
        if ($.fn.waypoint) {
            $('.skill').waypoint(function () {
                $('.progress .progress-bar').each(function () {
                    $(this).css("width", $(this).attr("aria-valuenow") + '%');
                });
            }, {offset: '80%'});
        }
        
        // Facts counter
        if ($.fn.counterUp) {
            $('[data-toggle="counter-up"]').counterUp({
                delay: 10,
                time: 2000
            });
        }
        
        // Testimonials carousel
        if ($.fn.owlCarousel) {
            $(".testimonial-carousel").owlCarousel({
                autoplay: true,
                smartSpeed: 1000,
                margin: 25,
                dots: false,
                loop: true,
                nav : true,
                navText : [
                    '<i class="bi bi-chevron-left"></i>',
                    '<i class="bi bi-chevron-right"></i>'
                ],
                responsive: {
                    0:{
                        items:1
                    },
                    992:{
                        items:2
                    }
                }
            });
        }
        
        // Portfolio isotope and filter
        if ($.fn.isotope) {
            var portfolioIsotope = $('.portfolio-container').isotope({
                itemSelector: '.portfolio-item',
                layoutMode: 'fitRows'
            });
            
            $('#portfolio-flters li').on('click', function () {
                $("#portfolio-flters li").removeClass('active');
                $(this).addClass('active');
                portfolioIsotope.isotope({filter: $(this).data('filter')});
            });
        }
    });
    
})(jQuery);
